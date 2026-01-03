# lift3d/models/act/act_actor.py
# -*- coding: utf-8 -*-
"""
Lift3D + ACT (DETR-VAE style) actor.

Key components (ACT/DETR-VAE):
1) Posterior q(z|a): encode action chunk -> (mu_q, logvar_q) -> z_q
2) Conditioned prior p(z|o): encode observation -> (mu_p, logvar_p) -> z_p
3) Training: use z_q, add KL(q||p) loss term
4) Inference: sample z from p(z|o) (NOT randn / zeros)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


Tensor = torch.Tensor


# -----------------------------
# VAE helpers
# -----------------------------
def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
    """z = mu + std * eps"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def kl_gaussian_qp(mu_q: Tensor, logvar_q: Tensor, mu_p: Tensor, logvar_p: Tensor) -> Tensor:
    """
    KL( N(mu_q, sigma_q) || N(mu_p, sigma_p) ), summed over latent dim.
    returns: [B]
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * torch.sum(
        (logvar_p - logvar_q) + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-12) - 1.0,
        dim=-1,
    )


@dataclass
class ActOutput:
    """Convenient structured output for training/eval."""
    actions: Tensor                 # [B, K, A]
    is_pad_hat: Tensor              # [B, K] (logits)

    # posterior q(z|a)
    mu: Optional[Tensor] = None     # [B, Z]
    logvar: Optional[Tensor] = None # [B, Z]

    # conditioned prior p(z|o)
    prior_mu: Optional[Tensor] = None       # [B, Z]
    prior_logvar: Optional[Tensor] = None   # [B, Z]

    # KL(q||p) (scalar) for training
    kl: Optional[Tensor] = None     # [] (mean over batch)

    # debug / visualization
    tokens: Optional[Tensor] = None        # [B, N, D] memory tokens after latent add
    global_feat: Optional[Tensor] = None   # [B, D]


class _LearnedPositionalEncoding(nn.Module):
    """Simple learned positional embedding for sequences (actions / queries)."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, T, D]
        return: [B, T, D]
        """
        _, t, _ = x.shape
        idx = torch.arange(t, device=x.device)
        return x + self.pos(idx)[None, :, :]


class Lift3DActActor(nn.Module):
    """
    DETR-style Transformer decoder head + ACT-style DETR-VAE latent,
    using Lift3D as perception backbone.

    Forward:
      inputs: images, point_clouds, robot_states, texts
      training supervision: actions [B,K,A], is_pad [B,K] bool

    Output:
      ActOutput if return_dict else actions_hat [B,K,A]
    """

    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        robot_state_dim: int,
        action_dim: int,

        # --- Hydra/backward-compat fields ---
        image_encoder: Optional[nn.Module] = None,
        fuse_method: str = "sum",
        rollout_mode: str = "replan",
        temporal_ensemble_coeff: float = 0.01,
        max_history: Optional[int] = None,

        # --- ACT core ---
        chunk_size: int = 50,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        latent_dim: int = 64,
        max_action_seq_len: int = 512,
        use_text: bool = False,
        return_dict: bool = True,

        **kwargs,
    ):
        super().__init__()
        # keep for compatibility
        self.image_encoder = image_encoder
        self.fuse_method = fuse_method
        self.rollout_mode = rollout_mode
        self.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        self.max_history = max_history
        _ = kwargs

        self.point_cloud_encoder = point_cloud_encoder
        self.robot_state_dim = int(robot_state_dim)
        self.action_dim = int(action_dim)

        self.K = int(chunk_size)
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.return_dict = bool(return_dict)

        # ---- Lift3D encoder -> tokens projection ----
        enc_dim = getattr(point_cloud_encoder, "feature_dim", None)
        if enc_dim is None:
            enc_dim = d_model
        self.enc_dim = int(enc_dim)

        self.enc_to_d = nn.Identity() if self.enc_dim == self.d_model else nn.Linear(self.enc_dim, self.d_model)

        # Robot state -> d_model (added into tokens / global)
        self.robot_to_d = nn.Linear(self.robot_state_dim, self.d_model)

        # ---- Posterior q(z|a): action chunk encoder ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.action_to_d = nn.Linear(self.action_dim, self.d_model)
        self.act_pos = _LearnedPositionalEncoding(max_action_seq_len + 1, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.cvae_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.latent_proj = nn.Linear(self.d_model, 2 * self.latent_dim)  # -> (mu_q, logvar_q)
        self.latent_out = nn.Linear(self.latent_dim, self.d_model)       # z -> d_model

        # ---- Conditioned prior p(z|o): from observation (global_d) ----
        # This is the missing key component in your current version.
        self.prior_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 2 * self.latent_dim),  # -> (mu_p, logvar_p)
        )

        # ---- DETR-style decoder head (memory tokens + latent -> chunk) ----
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

    # ---------------------------------------------------------------------
    # Lift3D encoder token extraction (your original robust implementation)
    # ---------------------------------------------------------------------
    def _lift3d_encode_tokens(
        self,
        images: Tensor,
        point_clouds: Tensor,
        robot_states: Tensor,
        texts: Any = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
          tokens_d: [B, N, D]
          global_d: [B, D]
        """
        enc = self.point_cloud_encoder

        # 0) One-time hook to capture point-cloud tokens from patch_embed
        if not hasattr(self, "_pc_tokens_hooked"):
            self._pc_tokens_hooked = True
            self._pc_tokens_cache: Dict[str, Tensor] = {}

            def _save_patch_tokens(_module, _inp, out):
                if "tokens" in self._pc_tokens_cache:
                    return
                try:
                    if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[1]):
                        feat = out[1]
                        if feat.dim() == 3:
                            if feat.shape[1] < feat.shape[2]:
                                tokens = feat.transpose(1, 2).contiguous()  # [B,N,C]
                            else:
                                tokens = feat.contiguous()  # already [B,N,C]
                            self._pc_tokens_cache["tokens"] = tokens
                except Exception:
                    return

            if hasattr(enc, "patch_embed") and isinstance(getattr(enc, "patch_embed"), nn.Module):
                enc.patch_embed.register_forward_hook(_save_patch_tokens)

        self._pc_tokens_cache.pop("tokens", None)

        # 1) Run encoder forward (try signatures)
        out: Any = None
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

        if out is None:
            for mname in ("encode_tokens", "forward_tokens", "forward_features"):
                if hasattr(enc, mname) and callable(getattr(enc, mname)):
                    try:
                        out = getattr(enc, mname)(point_clouds)  # type: ignore
                        break
                    except Exception:
                        continue

        if out is None:
            try:
                out = enc(point_clouds)
            except Exception:
                out = enc(images, point_clouds)  # type: ignore

        # 2) Normalize output
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

        # 3) Hook fallback
        if tokens is None:
            tokens = self._pc_tokens_cache.pop("tokens", None)

        # 4) Fix layout / derive
        if tokens is not None:
            if tokens.dim() != 3:
                tokens = None
            else:
                if tokens.shape[1] > tokens.shape[2]:
                    tokens = tokens.transpose(1, 2).contiguous()  # [B,N,C]

        if global_feat is None and tokens is not None:
            global_feat = tokens.mean(dim=1)

        if tokens is None and global_feat is not None:
            tokens = global_feat[:, None, :]

        if tokens is None or global_feat is None:
            raise RuntimeError(
                "Failed to obtain tokens/global from Lift3D encoder. "
                "Encoder may not expose patch tokens via patch_embed hook or returns unexpected format."
            )

        # 5) Project + fuse robot state
        tokens_d = self.enc_to_d(tokens)        # [B,N,D]
        global_d = self.enc_to_d(global_feat)   # [B,D]

        rs = self.robot_to_d(robot_states)      # [B,D]
        tokens_d = tokens_d + rs[:, None, :]
        global_d = global_d + rs

        return tokens_d, global_d

    # ---------------------------------------------------------------------
    # Posterior q(z|a)
    # ---------------------------------------------------------------------
    def _encode_latent(
        self,
        actions: Tensor,
        is_pad: Optional[Tensor],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        actions: [B,K,A]
        returns:
          latent_d: [B,D]
          mu_q/logvar_q: [B,Z]
        """
        bs = actions.shape[0]

        a = self.action_to_d(actions)                # [B,K,D]
        cls = self.cls_token.expand(bs, -1, -1)      # [B,1,D]
        x = torch.cat([cls, a], dim=1)               # [B,1+K,D]

        if is_pad is None:
            pad_mask = torch.zeros((bs, self.K + 1), dtype=torch.bool, device=device)
        else:
            if is_pad.dim() != 2 or is_pad.shape[1] != self.K:
                raise ValueError(f"is_pad must be [B,K], got {tuple(is_pad.shape)}")
            cls_pad = torch.zeros((bs, 1), dtype=torch.bool, device=device)
            pad_mask = torch.cat([cls_pad, is_pad.to(device)], dim=1)  # [B,1+K]

        x = self.act_pos(x)
        h = self.cvae_encoder(x, src_key_padding_mask=pad_mask)        # [B,1+K,D]
        h_cls = h[:, 0, :]                                             # [B,D]

        latent_info = self.latent_proj(h_cls)                          # [B,2Z]
        mu_q = latent_info[:, : self.latent_dim]
        logvar_q = latent_info[:, self.latent_dim :]

        z_q = reparametrize(mu_q, logvar_q)                             # [B,Z]
        latent_d = self.latent_out(z_q)                                 # [B,D]
        return latent_d, mu_q, logvar_q

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        images: Tensor,
        point_clouds: Tensor,
        robot_states: Tensor,
        texts: Any = None,
        actions: Optional[Tensor] = None,   # [B,K,A] during training
        is_pad: Optional[Tensor] = None,    # [B,K] bool during training
    ) -> Union[Tensor, ActOutput]:

        device = images.device if torch.is_tensor(images) else point_clouds.device

        # 1) Lift3D tokens as memory
        tokens_d, global_d = self._lift3d_encode_tokens(images, point_clouds, robot_states, texts)

        if not hasattr(self, "_printed"):
            print("tokens_d shape:", tokens_d.shape, "global_d shape:", global_d.shape)
            self._printed = True

        bs = tokens_d.shape[0]

        # 2) Conditioned prior p(z|o) from global_d
        prior_info = self.prior_proj(global_d)          # [B,2Z]
        mu_p = prior_info[:, : self.latent_dim]
        logvar_p = prior_info[:, self.latent_dim :]

        # 3) Latent sampling:
        #    - training: use posterior z_q, compute KL(q||p)
        #    - inference: sample z ~ p(z|o)
        kl = None
        if actions is not None:
            if actions.dim() != 3 or actions.shape[1] != self.K:
                raise ValueError(f"actions must be [B,K,A] with K={self.K}, got {tuple(actions.shape)}")

            latent_d, mu_q, logvar_q = self._encode_latent(actions, is_pad, device=device)
            kl = kl_gaussian_qp(mu_q, logvar_q, mu_p, logvar_p).mean()
            mu, logvar = mu_q, logvar_q
        else:
            z = reparametrize(mu_p, logvar_p)           # [B,Z]
            latent_d = self.latent_out(z)               # [B,D]
            mu = logvar = None

        # 4) Add latent to every memory token (ACT/DETR-VAE style)
        memory = tokens_d + latent_d[:, None, :]        # [B,N,D]

        # 5) DETR-style decoding with K queries
        q = self.query_embed.weight[None, :, :].expand(bs, -1, -1)     # [B,K,D]
        tgt = torch.zeros_like(q) + q

        hs = self.decoder(tgt=tgt, memory=memory)       # [B,K,D]

        actions_hat = self.action_head(hs)              # [B,K,A]
        is_pad_hat = self.is_pad_head(hs).squeeze(-1)   # [B,K]

        if self.return_dict:
            return ActOutput(
                actions=actions_hat,
                is_pad_hat=is_pad_hat,
                mu=mu,
                logvar=logvar,
                prior_mu=mu_p,
                prior_logvar=logvar_p,
                kl=kl,
                tokens=memory,
                global_feat=global_d,
            )
        return actions_hat