# lift3d/loss/act_vae_loss.py
# -*- coding: utf-8 -*-
"""
ACT-style VAE loss for chunk supervision.

This loss matches the original ACT training logic:
  loss = action_recon_loss + kl_weight * KL(mu, logvar)

- action_recon_loss: L1 (or SmoothL1) over actions, masked by ~is_pad
- KL: standard Gaussian KL divergence for VAE posterior vs N(0, I)

It is designed to work with different prediction formats:
1) preds is an ActOutput (from lift3d/models/act/act_actor.py):
      preds.actions   -> [B,K,A]
      preds.mu/logvar -> [B,Z]
      preds.is_pad_hat (optional) -> [B,K] logits
2) preds is a dict with keys: "actions", "mu", "logvar"
3) preds is a Tensor [B,K,A] (then KL term is 0)

Hydra usage example:
loss_func:
  _target_: lift3d.loss.act_vae_loss.act_vae_loss
  kl_weight: 0.01
  use_smooth_l1: false
  smooth_l1_beta: 1.0
  reduction: mean
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns:
      total_kld:         scalar tensor
      dimension_wise_kld: [Z]
      mean_kld:          scalar tensor
    """
    if mu.dim() > 2:
        mu = mu.view(mu.size(0), -1)
    if logvar.dim() > 2:
        logvar = logvar.view(logvar.size(0), -1)

    # klds: [B,Z]
    klds = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(dim=1).mean(dim=0, keepdim=True)          # [1]
    dimension_wise_kld = klds.mean(dim=0)                          # [Z]
    mean_kld = klds.mean(dim=1).mean(dim=0, keepdim=True)          # [1]
    return total_kld, dimension_wise_kld, mean_kld


def _unpack_preds(
    preds: Any,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Returns:
      actions_hat: [B,K,A]
      is_pad_hat:  [B,K] logits or None
      mu:          [B,Z] or None
      logvar:      [B,Z] or None
    """
    # Case 1: our ActOutput dataclass-like
    if hasattr(preds, "actions"):
        actions_hat = preds.actions
        is_pad_hat = getattr(preds, "is_pad_hat", None)
        mu = getattr(preds, "mu", None)
        logvar = getattr(preds, "logvar", None)
        return actions_hat, is_pad_hat, mu, logvar

    # Case 2: dict-like
    if isinstance(preds, dict):
        if "actions" in preds:
            actions_hat = preds["actions"]
        elif "actions_hat" in preds:
            actions_hat = preds["actions_hat"]
        else:
            raise ValueError("preds dict must contain key 'actions' or 'actions_hat'.")
        is_pad_hat = preds.get("is_pad_hat", None)
        mu = preds.get("mu", None)
        logvar = preds.get("logvar", None)
        return actions_hat, is_pad_hat, mu, logvar

    # Case 3: tensor only
    if torch.is_tensor(preds):
        return preds, None, None, None

    raise TypeError(f"Unsupported preds type: {type(preds)}")


def _masked_mean(x: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """
    x:    arbitrary shape
    mask: same broadcastable shape as x, bool (True keeps)
    """
    x = x * mask.to(dtype=x.dtype)
    denom = mask.to(dtype=x.dtype).sum().clamp_min(eps)
    return x.sum() / denom


def act_vae_loss(
    preds: Any,
    actions: Tensor,
    is_pad: Optional[Tensor] = None,
    kl_weight: float = 0.01,
    use_smooth_l1: bool = False,
    smooth_l1_beta: float = 1.0,
    reduction: str = "mean",
    # Optional: train is_pad prediction head (off by default to match most ACT codepaths)
    include_is_pad_loss: bool = False,
    pad_loss_weight: float = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    preds:
      - ActOutput / dict / Tensor
    actions:
      - [B,K,A]
    is_pad:
      - [B,K] bool, True means padded / invalid timestep
    """
    actions_hat, is_pad_hat, mu, logvar = _unpack_preds(preds)

    if actions_hat.dim() != 3 or actions.dim() != 3:
        raise ValueError(
            f"preds/actions must be 3D [B,K,A], got preds={tuple(actions_hat.shape)}, actions={tuple(actions.shape)}"
        )
    if actions_hat.shape != actions.shape:
        raise ValueError(
            f"Shape mismatch: preds={tuple(actions_hat.shape)} vs actions={tuple(actions.shape)}"
        )

    b, k, a = actions.shape

    # Default: no padding (all valid)
    if is_pad is None:
        valid = torch.ones((b, k), dtype=torch.bool, device=actions.device)
    else:
        if is_pad.dim() != 2 or is_pad.shape != (b, k):
            raise ValueError(f"is_pad must be [B,K]={b,k}, got {tuple(is_pad.shape)}")
        valid = ~is_pad.to(device=actions.device)

    # ---- reconstruction loss (masked) ----
    if use_smooth_l1:
        recon_all = F.smooth_l1_loss(
            actions_hat, actions, reduction="none", beta=float(smooth_l1_beta)
        )  # [B,K,A]
    else:
        recon_all = F.l1_loss(actions_hat, actions, reduction="none")  # [B,K,A]

    valid_3 = valid.unsqueeze(-1)  # [B,K,1]

    if reduction == "mean":
        recon = _masked_mean(recon_all, valid_3)
    elif reduction == "sum":
        recon = (recon_all * valid_3.to(recon_all.dtype)).sum()
    else:
        raise ValueError(f"Unsupported reduction='{reduction}', expected 'mean' or 'sum'.")

    # ---- KL loss ----
    if mu is not None and logvar is not None:
        total_kld, _, _ = kl_divergence(mu, logvar)  # [1]
        kl = total_kld.squeeze(0)
    else:
        kl = torch.zeros((), device=actions.device, dtype=actions.dtype)

    # ---- optional padding prediction loss ----
    pad_loss = torch.zeros((), device=actions.device, dtype=actions.dtype)
    if include_is_pad_loss:
        if is_pad is None:
            # If no labels, skip
            pad_loss = torch.zeros((), device=actions.device, dtype=actions.dtype)
        else:
            if is_pad_hat is None:
                raise ValueError("include_is_pad_loss=True but preds does not provide is_pad_hat.")
            if is_pad_hat.shape != (b, k):
                raise ValueError(f"is_pad_hat must be [B,K]={b,k}, got {tuple(is_pad_hat.shape)}")
            # target: 1 for padded, 0 for valid
            target = is_pad.to(dtype=actions.dtype, device=actions.device)
            pad_all = F.binary_cross_entropy_with_logits(is_pad_hat, target, reduction="none")  # [B,K]
            # Typically you can supervise pad everywhere (both valid and padded); we just mean over all K.
            if reduction == "mean":
                pad_loss = pad_all.mean()
            else:
                pad_loss = pad_all.sum()

    loss = recon + float(kl_weight) * kl + float(pad_loss_weight) * pad_loss

    loss_dict: Dict[str, Tensor] = {
        "loss": loss.detach(),
        "recon": recon.detach(),
        "kl": kl.detach(),
    }
    if include_is_pad_loss:
        loss_dict["pad_bce"] = pad_loss.detach()

    return loss, loss_dict