# lift3d/models/act/act_actor.py
# -*- coding: utf-8 -*-
"""
Lift3D + ACT (DETR-VAE style) actor.

Goal:
- Use Lift3D encoder (point cloud + RGB) as the *perception* backbone.
- Make the encoder return token / patch-map-like representations (tokens) for a Transformer decoder.
- Add the ACT-style CVAE latent (mu/logvar + reparameterization) conditioned on action chunks.

This file is written to be robust to different Lift3D encoder implementations:
- If the encoder supports returning tokens (e.g., return_tokens=True), we use them.
- Otherwise we fall back to repeating a global feature into "tokens" so the pipeline still runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass
class ActOutput:
    """Convenient structured output for training."""
    actions: Tensor                 # [B, K, A]
    is_pad_hat: Tensor              # [B, K] (logits)
    mu: Optional[Tensor]            # [B, Z]
    logvar: Optional[Tensor]        # [B, Z]
    tokens: Optional[Tensor]        # [B, N, D] memory tokens from Lift3D encoder
    global_feat: Optional[Tensor]   # [B, D]


def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
    """z = mu + std * eps"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


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
        b, t, _ = x.shape
        idx = torch.arange(t, device=x.device)
        return x + self.pos(idx)[None, :, :]


class Lift3DActActor(nn.Module):
    """
    A DETR-style Transformer decoder head with ACT-style CVAE latent,
    using Lift3D as the encoder.

    Forward I/O (matches your Lift3D training loop conventions):
      - Input observation (current step): images, point_clouds, robot_states, texts
      - Optional training supervision: actions [B,K,A], is_pad [B,K] (bool)

    Output:
      - If return_dict=True: ActOutput
      - Else: actions_hat [B,K,A]
    """

    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        robot_state_dim: int,
        action_dim: int,

        # --- Hydra/backward-compat fields (may exist in your agent yaml) ---
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

        # swallow any other unexpected keys safely
        **kwargs,
    ):
        super().__init__()
        # keep for compatibility (not used in current pipeline)
        self.image_encoder = image_encoder
        self.fuse_method = fuse_method
        self.rollout_mode = rollout_mode
        self.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        self.max_history = max_history

        # (ignore kwargs on purpose)
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
            # If your encoder doesn't expose feature_dim, assume it already matches d_model.
            enc_dim = d_model
        self.enc_dim = int(enc_dim)

        self.enc_to_d = nn.Identity() if self.enc_dim == self.d_model else nn.Linear(self.enc_dim, self.d_model)

        # Robot state -> d_model (added into tokens / global)
        self.robot_to_d = nn.Linear(self.robot_state_dim, self.d_model)

        # ---- ACT CVAE encoder (actions -> mu/logvar) ----
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
            batch_first=True,   # [B,T,D]
            norm_first=False,
        )
        self.cvae_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.latent_proj = nn.Linear(self.d_model, 2 * self.latent_dim)   # -> (mu, logvar)
        self.latent_out = nn.Linear(self.latent_dim, self.d_model)

        # ---- DETR-style decoder head (tokens + latent -> action chunk) ----
        self.query_embed = nn.Embedding(self.K, self.d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,   # [B,T,D]
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.action_head = nn.Linear(self.d_model, self.action_dim)
        self.is_pad_head = nn.Linear(self.d_model, 1)

        # For stability
        nn.init.constant_(self.is_pad_head.bias, 0.0)

    # ---------------------------------------------------------------------
    # Lift3D encoder token extraction (robust to different implementations)
    # ---------------------------------------------------------------------
   
    def _lift3d_encode_tokens(
        self,
        images: Tensor,
        point_clouds: Tensor,
        robot_states: Tensor,
        texts: Any = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode observation with a Lift3D-style point-cloud encoder and return:

        tokens_d: [B, N, D]  memory tokens (for Transformer decoder)
        global_d: [B, D]     global feature

        This function is robust across different Lift3D encoder implementations:

        Priority order:
        1) Encoder natively returns tokens/global (tuple or dict) -> use them.
        2) If encoder returns only a global embedding [B,C], we *hook* into
            enc.patch_embed to capture point-cloud tokens during the same forward.
        3) If still no tokens, fall back to a single-token sequence from global.

        Notes:
        - Many point encoders produce patch tokens inside `patch_embed` as
            (xyz, feat) where feat is [B, C, N]. We convert to [B, N, C].
        - Hooks are registered once and capture tokens without printing.
        """
        enc = self.point_cloud_encoder
        device = point_clouds.device

        # ------------------------------------------------------------
        # 0) One-time hook to capture point-cloud tokens from patch_embed
        # ------------------------------------------------------------
        if not hasattr(self, "_pc_tokens_hooked"):
            self._pc_tokens_hooked = True
            self._pc_tokens_cache: Dict[str, Tensor] = {}

            def _save_patch_tokens(_module, _inp, out):
                """
                Capture patch tokens from patch_embed output.
                Common format: (xyz: [B,N,3], feat: [B,C,N]) or (xyz, feat, ...)
                """
                # Save only once per forward to avoid overwriting / extra overhead.
                if "tokens" in self._pc_tokens_cache:
                    return

                try:
                    if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[1]):
                        feat = out[1]
                        # Expect feat as [B, C, N] or [B, N, C]
                        if feat.dim() == 3:
                            if feat.shape[1] < feat.shape[2]:
                                # Likely [B, C, N] -> [B, N, C]
                                tokens = feat.transpose(1, 2).contiguous()
                            else:
                                # Already [B, N, C]
                                tokens = feat.contiguous()
                            self._pc_tokens_cache["tokens"] = tokens
                except Exception:
                    # Never crash training because of token capture.
                    return

            # Register the hook if patch_embed exists.
            if hasattr(enc, "patch_embed") and isinstance(getattr(enc, "patch_embed"), nn.Module):
                enc.patch_embed.register_forward_hook(_save_patch_tokens)

        # Clear any stale tokens (e.g., from a previous step if something went wrong).
        self._pc_tokens_cache.pop("tokens", None)

        # ------------------------------------------------------------
        # 1) Run encoder forward (try multiple signatures)
        # ------------------------------------------------------------
        out: Any = None

        # Pattern A: kwargs with return_tokens (if supported by your encoder)
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

        # Pattern B: explicit methods
        if out is None:
            for mname in ("encode_tokens", "forward_tokens", "forward_features"):
                if hasattr(enc, mname) and callable(getattr(enc, mname)):
                    try:
                        out = getattr(enc, mname)(point_clouds)  # type: ignore
                        break
                    except Exception:
                        continue

        # Pattern C: default forward
        if out is None:
            try:
                out = enc(point_clouds)  # common in your current setup
            except Exception:
                out = enc(images, point_clouds)  # type: ignore

        # ------------------------------------------------------------
        # 2) Normalize encoder output into (tokens, global)
        # ------------------------------------------------------------
        tokens: Optional[Tensor] = None
        global_feat: Optional[Tensor] = None

        if isinstance(out, (tuple, list)):
            # Heuristic:
            # - tokens: 3D tensor [B,N,C] or [B,C,N]
            # - global: 2D tensor [B,C]
            for x in out:
                if torch.is_tensor(x) and x.dim() == 3 and tokens is None:
                    tokens = x
                elif torch.is_tensor(x) and x.dim() == 2 and global_feat is None:
                    global_feat = x

        elif isinstance(out, dict):
            # Common keys for tokens
            for k in ("tokens", "patch_tokens", "token", "patch_map", "patch"):
                if k in out and torch.is_tensor(out[k]) and out[k].dim() == 3:
                    tokens = out[k]
                    break
            # Common keys for global
            for k in ("global", "feat", "feature", "embedding", "emb"):
                if k in out and torch.is_tensor(out[k]) and out[k].dim() == 2:
                    global_feat = out[k]
                    break
            # If only tokens exist, derive global by mean-pooling
            if global_feat is None and tokens is not None:
                # Convert token layout first (below), then mean over N.
                pass

        else:
            # Tensor output: usually global [B,C]
            if torch.is_tensor(out) and out.dim() == 2:
                global_feat = out

        # ------------------------------------------------------------
        # 3) If tokens weren't returned, try grabbing from hook cache
        # ------------------------------------------------------------
        if tokens is None:
            tokens = self._pc_tokens_cache.pop("tokens", None)

        # ------------------------------------------------------------
        # 4) Fix token layout and derive missing global/tokens if needed
        # ------------------------------------------------------------
        if tokens is not None:
            # Accept either [B,N,C] or [B,C,N], convert to [B,N,C]
            if tokens.dim() != 3:
                tokens = None
            else:
                # If shape looks like [B,C,N] (common), swap.
                # We cannot know C vs N for sure, but in your logs C=768, N=128.
                if tokens.shape[1] > tokens.shape[2]:
                    tokens = tokens.transpose(1, 2).contiguous()

        if global_feat is None and tokens is not None:
            global_feat = tokens.mean(dim=1)

        if tokens is None and global_feat is not None:
            # Fallback: single-token sequence
            tokens = global_feat[:, None, :]

        if tokens is None or global_feat is None:
            raise RuntimeError(
                "Failed to obtain tokens/global from the Lift3D encoder. "
                "Your encoder might not expose patch tokens via patch_embed hook, "
                "or it returns an unexpected format."
            )

        # ------------------------------------------------------------
        # 5) Project to d_model and fuse robot state
        # ------------------------------------------------------------
        tokens_d = self.enc_to_d(tokens)        # [B,N,D]
        global_d = self.enc_to_d(global_feat)   # [B,D]

        rs = self.robot_to_d(robot_states)      # [B,D]
        tokens_d = tokens_d + rs[:, None, :]
        global_d = global_d + rs

        return tokens_d, global_d
        
        # ---------------------------------------------------------------------
        # CVAE latent from action chunk
        # ---------------------------------------------------------------------
    def _encode_latent(
        self,
        actions: Optional[Tensor],
        is_pad: Optional[Tensor],
        device: torch.device,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Returns:
          latent_d: [B,D]  (latent in model space)
          mu/logvar: [B,Z] or None in inference
        """
        if actions is None:
            # Inference prior: z = 0
            # Use zeros -> latent_out -> [B,D]
            # We need batch size, so caller will set it by creating zeros with bs.
            raise RuntimeError("Internal error: actions=None path should be handled with batch size.")
        bs = actions.shape[0]

        # actions: [B,K,A] -> [B,K,D]
        a = self.action_to_d(actions)

        # prepend CLS token
        cls = self.cls_token.expand(bs, -1, -1)  # [B,1,D]
        x = torch.cat([cls, a], dim=1)           # [B,1+K,D]

        # key padding mask: True means "ignore"
        if is_pad is None:
            pad_mask = torch.zeros((bs, self.K + 1), dtype=torch.bool, device=device)
        else:
            if is_pad.dim() != 2 or is_pad.shape[1] != self.K:
                raise ValueError(f"is_pad must be [B,K], got {tuple(is_pad.shape)}")
            cls_pad = torch.zeros((bs, 1), dtype=torch.bool, device=device)
            pad_mask = torch.cat([cls_pad, is_pad.to(device)], dim=1)

        x = self.act_pos(x)
        h = self.cvae_encoder(x, src_key_padding_mask=pad_mask)  # [B,1+K,D]
        h_cls = h[:, 0, :]                                       # [B,D]

        latent_info = self.latent_proj(h_cls)                    # [B,2Z]
        mu = latent_info[:, : self.latent_dim]
        logvar = latent_info[:, self.latent_dim :]

        z = reparametrize(mu, logvar)                            # [B,Z]
        latent_d = self.latent_out(z)                            # [B,D]
        return latent_d, mu, logvar

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
        """
        If actions is provided -> training-style forward (CVAE posterior, returns mu/logvar).
        If actions is None -> inference-style forward (prior z=0).
        """
        device = images.device if torch.is_tensor(images) else point_clouds.device

        # 1) Lift3D tokens as memory
        tokens_d, global_d = self._lift3d_encode_tokens(images, point_clouds, robot_states, texts)
        # tokens_d: [B,N,D]

        if not hasattr(self, "_printed"):
            print("tokens_d shape:", tokens_d.shape, "global_d shape:", global_d.shape)
            self._printed = True

        bs = tokens_d.shape[0]

        # 2) Latent conditioning
        if actions is not None:
            if actions.dim() != 3 or actions.shape[1] != self.K:
                raise ValueError(f"actions must be [B,K,A] with K={self.K}, got {tuple(actions.shape)}")
            latent_d, mu, logvar = self._encode_latent(actions, is_pad, device=device)
        else:
            mu = logvar = None
            z = torch.zeros((bs, self.latent_dim), device=device, dtype=tokens_d.dtype)
            latent_d = self.latent_out(z)  # [B,D]

        # Add latent to every memory token (matches the original ACT code style)
        memory = tokens_d + latent_d[:, None, :]  # [B,N,D]

        # 3) DETR-style decoding with K queries
        # query embedding -> [B,K,D]
        q = self.query_embed.weight[None, :, :].expand(bs, -1, -1)
        tgt = torch.zeros_like(q) + q

        # nn.TransformerDecoder expects:
        #   tgt:    [B,K,D]
        #   memory: [B,N,D]
        hs = self.decoder(tgt=tgt, memory=memory)  # [B,K,D]

        actions_hat = self.action_head(hs)                # [B,K,A]
        is_pad_hat = self.is_pad_head(hs).squeeze(-1)     # [B,K] logits

        if self.return_dict:
            return ActOutput(
                actions=actions_hat,
                is_pad_hat=is_pad_hat,
                mu=mu,
                logvar=logvar,
                tokens=memory,
                global_feat=global_d,
            )
        return actions_hat