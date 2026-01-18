# lift3d/models/act_actor_normal.py
# -*- coding: utf-8 -*-
"""
Lift3D + ACT (DETR-VAE style) actor (with conditional prior) AND Auto-Normalization.

Key components added:
1) Perception encoder -> memory tokens for Transformer decoder (DETR-style)
2) Posterior q(z | actions, obs) (ACT-style, TransformerEncoder over action chunk + obs token)
3) Conditional prior p(z | obs)  (MLP over global obs feature)
4) Train with KL(q||p); Inference uses prior (mean or sample)
5) [NEW] Auto-Normalization: Integrated Z-Score normalization for stable surgical fine-tuning.

Robust token extraction:
- If Lift3D encoder returns tokens/global -> use them.
- Else try forward-hook on encoder.patch_embed to capture tokens.
- Else fallback to single token from global feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


Tensor = torch.Tensor


@dataclass
class ActOutput:
    actions: Tensor                   # [B, K, A] (Predicted Actions in REAL SCALE)
    is_pad_hat: Optional[Tensor]      # [B, K] logits (optional)
    mu: Optional[Tensor]              # posterior mu   [B, Z]
    logvar: Optional[Tensor]          # posterior logvar [B, Z]
    prior_mu: Optional[Tensor]        # prior mu       [B, Z]
    prior_logvar: Optional[Tensor]    # prior logvar   [B, Z]
    tokens: Optional[Tensor]          # [B, N, D] (memory actually fed to decoder)
    global_feat: Optional[Tensor]     # [B, D]


def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


class _LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding for sequences [B,T,D]."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        idx = torch.arange(t, device=x.device)
        return x + self.pos(idx)[None, :, :]


class Lift3DActActor(nn.Module):
    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        robot_state_dim: int,
        action_dim: int,

        # --- compatibility fields (ignored / kept for hydra configs) ---
        image_encoder: Optional[nn.Module] = None,
        fuse_method: str = "sum",
        rollout_mode: str = "replan",
        temporal_ensemble_coeff: float = 0.01,
        max_history: Optional[int] = None,

        # --- ACT/DETR-VAE core ---
        chunk_size: int = 50,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,

        latent_dim: int = 64,
        max_action_seq_len: int = 512,

        # behavior
        return_dict: bool = True,
        sample_prior: bool = False,   # inference: sample from prior or use mean
        **kwargs,
    ):
        super().__init__()
        _ = kwargs  # swallow unexpected hydra keys

        self.image_encoder = image_encoder
        self.fuse_method = fuse_method
        self.rollout_mode = rollout_mode
        self.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        self.max_history = max_history

        self.point_cloud_encoder = point_cloud_encoder
        self.robot_state_dim = int(robot_state_dim)
        self.action_dim = int(action_dim)

        self.K = int(chunk_size)
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.return_dict = bool(return_dict)
        self.sample_prior = bool(sample_prior)

        # ----- encoder feature dim -> d_model -----
        enc_dim = getattr(point_cloud_encoder, "feature_dim", None)
        if enc_dim is None:
            enc_dim = d_model
        self.enc_dim = int(enc_dim)
        self.enc_to_d = nn.Identity() if self.enc_dim == self.d_model else nn.Linear(self.enc_dim, self.d_model)

        # robot state -> d_model
        self.robot_to_d = nn.Linear(self.robot_state_dim, self.d_model)

        # ----- posterior q(z|a,obs): TransformerEncoder over [CLS, OBS, a1..aK] -----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.action_to_d = nn.Linear(self.action_dim, self.d_model)
        # +2 for CLS + OBS
        self.act_pos = _LearnedPositionalEncoding(max_action_seq_len + 2, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.posterior_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.posterior_proj = nn.Linear(self.d_model, 2 * self.latent_dim)  # -> mu/logvar

        # ----- conditional prior p(z|obs): MLP(global_obs) -----
        self.prior_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 2 * self.latent_dim),
        )

        # latent z -> model space d_model
        self.latent_out = nn.Linear(self.latent_dim, self.d_model)

        # ----- DETR-style decoder head -----
        self.query_embed = nn.Embedding(self.K, self.d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.action_head = nn.Linear(self.d_model, self.action_dim)
        self.is_pad_head = nn.Linear(self.d_model, 1)
        nn.init.constant_(self.is_pad_head.bias, 0.0)

        # ---- token hook state ----
        self._pc_tokens_hooked = False
        self._pc_tokens_cache: Dict[str, Tensor] = {}

        # =========================================================================
        # [NEW] Auto-Normalization Buffers
        # =========================================================================
        # These will be saved in the .pth file but not trained via gradient descent.
        self.register_buffer('action_mean', torch.zeros(self.action_dim))
        self.register_buffer('action_std', torch.ones(self.action_dim))
        self.register_buffer('qpos_mean', torch.zeros(self.robot_state_dim))
        self.register_buffer('qpos_std', torch.ones(self.robot_state_dim))
        self.register_buffer('is_norm_initialized', torch.tensor(0, dtype=torch.bool))

    # ---------------------------------------------------------------------
    # [NEW] Normalization Methods
    # ---------------------------------------------------------------------
    def set_norm_stats(self, action_mean, action_std, qpos_mean, qpos_std):
        """Called by training script to inject dataset statistics."""
        # Safety clip to avoid div by zero
        action_std = torch.clip(action_std, min=1e-5)
        qpos_std = torch.clip(qpos_std, min=1e-5)

        self.action_mean.data = action_mean
        self.action_std.data = action_std
        self.qpos_mean.data = qpos_mean
        self.qpos_std.data = qpos_std
        self.is_norm_initialized.data = torch.tensor(1, dtype=torch.bool, device=self.is_norm_initialized.device)
        print("ACTPolicy: Normalization stats updated and locked into model buffers.")

    def normalize_qpos(self, qpos: Tensor) -> Tensor:
        if not self.is_norm_initialized:
            return qpos
        return (qpos - self.qpos_mean) / self.qpos_std

    def normalize_actions(self, actions: Tensor) -> Tensor:
        if not self.is_norm_initialized:
            return actions
        return (actions - self.action_mean) / self.action_std

    def unnormalize_actions(self, actions_norm: Tensor) -> Tensor:
        if not self.is_norm_initialized:
            return actions_norm
        return actions_norm * self.action_std + self.action_mean

    # ---------------------------------------------------------------------
    # Lift3D encoder token extraction
    # ---------------------------------------------------------------------
    def _maybe_register_pc_hook(self):
        if self._pc_tokens_hooked:
            return
        self._pc_tokens_hooked = True

        enc = self.point_cloud_encoder

        def _save_patch_tokens(_module, _inp, out):
            # cache once per forward
            if "tokens" in self._pc_tokens_cache:
                return
            try:
                if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[1]):
                    feat = out[1]
                    if feat.dim() == 3:
                        # [B,C,N] -> [B,N,C] (common)
                        if feat.shape[1] < feat.shape[2]:
                            tokens = feat.transpose(1, 2).contiguous()
                        else:
                            tokens = feat.contiguous()
                        self._pc_tokens_cache["tokens"] = tokens
            except Exception:
                return

        if hasattr(enc, "patch_embed") and isinstance(getattr(enc, "patch_embed"), nn.Module):
            enc.patch_embed.register_forward_hook(_save_patch_tokens)

    def _lift3d_encode_tokens(
        self,
        images: Tensor,
        point_clouds: Tensor,
        robot_states: Tensor,
        texts: Any = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
          tokens_d: [B,N,D]
          global_d: [B,D]
        """
        self._maybe_register_pc_hook()
        self._pc_tokens_cache.pop("tokens", None)

        enc = self.point_cloud_encoder
        out: Any = None

        # try kwargs signatures
        for kwargs in (
            dict(point_clouds=point_clouds, images=images, return_tokens=True),
            dict(point_clouds=point_clouds, return_tokens=True),
            dict(x=point_clouds, return_tokens=True),
        ):
            try:
                out = enc(**kwargs)  # type: ignore
                break
            except TypeError:
                continue
            except Exception:
                continue

        # try explicit token methods
        if out is None:
            for mname in ("encode_tokens", "forward_tokens", "forward_features"):
                if hasattr(enc, mname) and callable(getattr(enc, mname)):
                    try:
                        out = getattr(enc, mname)(point_clouds)  # type: ignore
                        break
                    except Exception:
                        continue

        # default forward
        if out is None:
            try:
                out = enc(point_clouds)
            except Exception:
                out = enc(images, point_clouds)  # type: ignore

        tokens: Optional[Tensor] = None
        global_feat: Optional[Tensor] = None

        if isinstance(out, (tuple, list)):
            for x in out:
                if torch.is_tensor(x) and x.dim() == 3 and tokens is None:
                    tokens = x
                elif torch.is_tensor(x) and x.dim() == 2 and global_feat is None:
                    global_feat = x
        elif isinstance(out, dict):
            for k in ("tokens", "patch_tokens", "token", "patch_map", "patch"):
                if k in out and torch.is_tensor(out[k]) and out[k].dim() == 3:
                    tokens = out[k]
                    break
            for k in ("global", "feat", "feature", "embedding", "emb"):
                if k in out and torch.is_tensor(out[k]) and out[k].dim() == 2:
                    global_feat = out[k]
                    break
        else:
            if torch.is_tensor(out) and out.dim() == 2:
                global_feat = out

        if tokens is None:
            tokens = self._pc_tokens_cache.pop("tokens", None)

        # normalize token layout to [B,N,C]
        if tokens is not None:
            if tokens.dim() != 3:
                tokens = None
            else:
                # if looks like [B,C,N] -> transpose
                if tokens.shape[1] > tokens.shape[2]:
                    tokens = tokens.transpose(1, 2).contiguous()

        if global_feat is None and tokens is not None:
            global_feat = tokens.mean(dim=1)

        if tokens is None and global_feat is not None:
            tokens = global_feat[:, None, :]

        if tokens is None or global_feat is None:
            raise RuntimeError("Failed to get tokens/global from Lift3D encoder output or hook.")

        tokens_d = self.enc_to_d(tokens)
        global_d = self.enc_to_d(global_feat)

        # [NEW] Use Normalized robot states for embedding injection
        # This ensures modalities (Point Cloud vs Proprio) are on similar scales
        rs = self.robot_to_d(robot_states)  # [B,D] (Note: robot_states should be normalized before this call)
        
        tokens_d = tokens_d + rs[:, None, :]
        global_d = global_d + rs
        return tokens_d, global_d

    # ---------------------------------------------------------------------
    # posterior q(z|a,obs) and prior p(z|obs)
    # ---------------------------------------------------------------------
    def _posterior(
        self,
        global_d: Tensor,
        actions: Tensor,
        is_pad: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        global_d: [B,D]
        actions:  [B,K,A]
        is_pad:   [B,K] bool (True padded)
        """
        bs = actions.shape[0]

        a = self.action_to_d(actions)             # [B,K,D]
        cls = self.cls_token.expand(bs, -1, -1)   # [B,1,D]
        obs = global_d[:, None, :]                # [B,1,D]
        x = torch.cat([cls, obs, a], dim=1)       # [B,2+K,D]

        if is_pad is None:
            pad_mask = torch.zeros((bs, 2 + self.K), dtype=torch.bool, device=actions.device)
        else:
            if is_pad.shape != (bs, self.K):
                raise ValueError(f"is_pad must be [B,K], got {tuple(is_pad.shape)}")
            head = torch.zeros((bs, 2), dtype=torch.bool, device=actions.device)
            pad_mask = torch.cat([head, is_pad.to(device=actions.device)], dim=1)

        x = self.act_pos(x)
        h = self.posterior_encoder(x, src_key_padding_mask=pad_mask)  # [B,2+K,D]
        h_cls = h[:, 0, :]                                            # [B,D]
        stats = self.posterior_proj(h_cls)                            # [B,2Z]
        mu = stats[:, : self.latent_dim]
        logvar = stats[:, self.latent_dim :]
        return mu, logvar

    def _prior(self, global_d: Tensor) -> Tuple[Tensor, Tensor]:
        stats = self.prior_net(global_d)  # [B,2Z]
        mu = stats[:, : self.latent_dim]
        logvar = stats[:, self.latent_dim :]
        return mu, logvar

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        images: Tensor,
        point_clouds: Tensor,
        robot_states: Tensor,
        texts: Any = None,
        actions: Optional[Tensor] = None,   # [B,K,A] in training
        is_pad: Optional[Tensor] = None,    # [B,K] bool
    ) -> Union[Tensor, ActOutput]:
        """
        Full Forward pass with Auto-Normalization.
        Inputs (robot_states, actions) are expected in RAW PHYSICAL SCALE (e.g. mm).
        They are normalized internally before passing to the network.
        Outputs are un-normalized back to RAW PHYSICAL SCALE.
        """
        
        # 1. Normalize Robot State (Proprioception)
        # This fixes the scale mismatch between PointCloud (Meters) and Proprio (Millimeters)
        robot_states_norm = self.normalize_qpos(robot_states)

        # 2. Encode Perceptual Features using Normalized Proprio
        tokens_d, global_d = self._lift3d_encode_tokens(images, point_clouds, robot_states_norm, texts)
        bs = tokens_d.shape[0]

        # prior always available
        prior_mu, prior_logvar = self._prior(global_d)

        mu, logvar = None, None
        
        # 3. Handle Actions (Training vs Inference)
        if actions is not None:
            # Training Mode
            if actions.dim() != 3 or actions.shape[1] != self.K:
                raise ValueError(f"actions must be [B,K,A] with K={self.K}, got {tuple(actions.shape)}")
            
            # Normalize GT Actions for VAE posterior
            # The VAE learns to reconstruct Normalized actions (N(0,1))
            actions_norm = self.normalize_actions(actions)
            
            mu, logvar = self._posterior(global_d, actions_norm, is_pad)
            z = reparametrize(mu, logvar)   # posterior sample
        else:
            # Inference Mode
            if self.sample_prior:
                z = reparametrize(prior_mu, prior_logvar)
            else:
                z = prior_mu

        # 4. Decoder / Policy Head
        latent_d = self.latent_out(z)       # [B,D]
        memory = tokens_d + latent_d[:, None, :]  # [B,N,D]

        # DETR-style queries
        q = self.query_embed.weight[None, :, :].expand(bs, -1, -1)  # [B,K,D]
        tgt = q

        hs = self.decoder(tgt=tgt, memory=memory)  # [B,K,D]

        actions_hat_norm = self.action_head(hs)          # [B,K,A] (Normalized Space)
        is_pad_hat = self.is_pad_head(hs).squeeze(-1)    # [B,K]

        # 5. Un-normalize Output
        # Return predictions in Real Physical Scale so external Loss/Metric functions work as expected.
        # Note: Ideally loss should be computed in Normalized space, but un-normalizing here ensures
        # compatibility with existing code that expects raw units.
        if actions is not None:
            # 【训练模式】
            # 直接返回归一化的预测值！
            # 这样 Loss 就会在归一化空间计算，Position 和 Rotation 权重自动平衡。
            if self.return_dict:
                return ActOutput(
                    actions=actions_hat_norm, # <--- 注意这里，返回的是 norm
                    is_pad_hat=is_pad_hat,
                    mu=mu,
                    logvar=logvar,
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                    tokens=memory,
                    global_feat=global_d,
                )
            return actions_hat_norm
        else:
            # 【推理模式】
            # 必须反归一化，因为机器人需要真实的毫米/弧度
        actions_hat = self.unnormalize_actions(actions_hat_norm)

        if self.return_dict:
            return ActOutput(
                actions=actions_hat,
                is_pad_hat=is_pad_hat,
                mu=mu,
                logvar=logvar,
                prior_mu=prior_mu,
                prior_logvar=prior_logvar,
                tokens=memory,
                global_feat=global_d,
            )
        return actions_hat