# lift3d/loss/act_vae_loss.py
# -*- coding: utf-8 -*-
"""
ACT/DETR-VAE loss for chunk supervision.

- recon: L1 or SmoothL1 over actions, masked by ~is_pad
- KL:
  - if prior_mu/prior_logvar exist -> KL(q||p)
  - else -> KL(q||N(0,I)) (classic ACT)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _masked_mean(x: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    x = x * mask.to(dtype=x.dtype)
    denom = mask.to(dtype=x.dtype).sum().clamp_min(eps)
    return x.sum() / denom


def kl_q_to_std_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    # klds: [B,Z]
    klds = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(dim=1).mean()


def kl_diag_gaussian_qp(mu_q: Tensor, logvar_q: Tensor, mu_p: Tensor, logvar_p: Tensor) -> Tensor:
    """
    KL( N(mu_q, diag(exp(logvar_q))) || N(mu_p, diag(exp(logvar_p))) )
    returns scalar averaged over batch
    """
    # [B,Z]
    var_q = logvar_q.exp()
    var_p = logvar_p.exp().clamp_min(1e-8)

    term1 = (logvar_p - logvar_q)  # log(var_p/var_q)
    term2 = (var_q + (mu_q - mu_p).pow(2)) / var_p
    kl = 0.5 * (term1 + term2 - 1.0)
    return kl.sum(dim=1).mean()


def _unpack_preds(
    preds: Any,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Returns:
      actions_hat: [B,K,A]
      is_pad_hat:  [B,K] logits or None
      mu/logvar: posterior [B,Z] or None
      prior_mu/prior_logvar: [B,Z] or None
    """
    # ActOutput-like
    if hasattr(preds, "actions"):
        actions_hat = preds.actions
        is_pad_hat = getattr(preds, "is_pad_hat", None)
        mu = getattr(preds, "mu", None)
        logvar = getattr(preds, "logvar", None)
        prior_mu = getattr(preds, "prior_mu", None)
        prior_logvar = getattr(preds, "prior_logvar", None)
        return actions_hat, is_pad_hat, mu, logvar, prior_mu, prior_logvar

    # dict-like
    if isinstance(preds, dict):
        if "actions" in preds:
            actions_hat = preds["actions"]
        elif "actions_hat" in preds:
            actions_hat = preds["actions_hat"]
        else:
            raise ValueError("preds dict must contain 'actions' or 'actions_hat'.")

        is_pad_hat = preds.get("is_pad_hat", None)
        mu = preds.get("mu", None)
        logvar = preds.get("logvar", None)
        prior_mu = preds.get("prior_mu", None)
        prior_logvar = preds.get("prior_logvar", None)
        return actions_hat, is_pad_hat, mu, logvar, prior_mu, prior_logvar

    # tensor-only
    if torch.is_tensor(preds):
        return preds, None, None, None, None, None

    raise TypeError(f"Unsupported preds type: {type(preds)}")


def act_vae_loss(
    preds: Any,
    actions: Tensor,
    is_pad: Optional[Tensor] = None,
    kl_weight: float = 0.01,
    use_smooth_l1: bool = False,
    smooth_l1_beta: float = 1.0,
    reduction: str = "mean",
    include_is_pad_loss: bool = False,
    pad_loss_weight: float = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    actions_hat, is_pad_hat, mu, logvar, prior_mu, prior_logvar = _unpack_preds(preds)

    if actions_hat.dim() != 3 or actions.dim() != 3:
        raise ValueError(f"preds/actions must be [B,K,A], got preds={tuple(actions_hat.shape)}, actions={tuple(actions.shape)}")
    if actions_hat.shape != actions.shape:
        raise ValueError(f"Shape mismatch: preds={tuple(actions_hat.shape)} vs actions={tuple(actions.shape)}")

    b, k, a = actions.shape

    # mask
    if is_pad is None:
        valid = torch.ones((b, k), dtype=torch.bool, device=actions.device)
    else:
        if is_pad.shape != (b, k):
            raise ValueError(f"is_pad must be [B,K]={b,k}, got {tuple(is_pad.shape)}")
        valid = ~is_pad.to(device=actions.device)

    # recon
    if use_smooth_l1:
        recon_all = F.smooth_l1_loss(actions_hat, actions, reduction="none", beta=float(smooth_l1_beta))
    else:
        recon_all = F.l1_loss(actions_hat, actions, reduction="none")

    valid_3 = valid.unsqueeze(-1)  # [B,K,1]
    if reduction == "mean":
        recon = _masked_mean(recon_all, valid_3)
    elif reduction == "sum":
        recon = (recon_all * valid_3.to(recon_all.dtype)).sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    # KL
    if (mu is not None) and (logvar is not None):
        if (prior_mu is not None) and (prior_logvar is not None):
            kl = kl_diag_gaussian_qp(mu, logvar, prior_mu, prior_logvar)
        else:
            kl = kl_q_to_std_normal(mu, logvar)
    else:
        kl = torch.zeros((), device=actions.device, dtype=actions.dtype)

    # optional pad head loss
    pad_loss = torch.zeros((), device=actions.device, dtype=actions.dtype)
    if include_is_pad_loss and (is_pad is not None):
        if is_pad_hat is None:
            raise ValueError("include_is_pad_loss=True but preds has no is_pad_hat.")
        if is_pad_hat.shape != (b, k):
            raise ValueError(f"is_pad_hat must be [B,K]={b,k}, got {tuple(is_pad_hat.shape)}")
        target = is_pad.to(dtype=actions.dtype, device=actions.device)  # 1 for padded
        pad_all = F.binary_cross_entropy_with_logits(is_pad_hat, target, reduction="none")  # [B,K]
        pad_loss = pad_all.mean() if reduction == "mean" else pad_all.sum()

    loss = recon + float(kl_weight) * kl + float(pad_loss_weight) * pad_loss

    loss_dict: Dict[str, Tensor] = {
        "loss": loss.detach(),
        "recon": recon.detach(),
        "kl": kl.detach(),
    }
    if include_is_pad_loss:
        loss_dict["pad_bce"] = pad_loss.detach()

    return loss, loss_dict