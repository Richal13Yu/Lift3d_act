# lift3d/loss/act_chunk_loss.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def act_chunk_l1_loss(
    preds: torch.Tensor,
    actions: torch.Tensor,
    is_pad: Optional[torch.Tensor] = None,
    *,
    use_smooth_l1: bool = False,
    smooth_l1_beta: float = 1.0,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Masked chunk regression loss for ACT-style training.

    Expected shapes:
      preds:   [B, K, A]  (model predicted action chunk)
      actions: [B, K, A]  (ground-truth action chunk)
      is_pad:  [B, K]     (bool) True means padded/invalid step

    Returns:
      (loss, loss_dict) so it fits your train loop pattern:
        loss_result = call(loss_func, preds, actions, is_pad=...)
        if tuple -> (loss, dict)

    Why this works:
      - ACT uses chunk supervision a[t:t+K]; near episode end we pad.
      - We multiply per-step loss by (~is_pad) so padded steps don't contribute.
    """
    if preds.dim() != 3 or actions.dim() != 3:
        raise ValueError(f"preds/actions must be 3D [B,K,A], got preds={preds.shape}, actions={actions.shape}")
    if preds.shape != actions.shape:
        raise ValueError(f"preds and actions shapes must match, got preds={preds.shape}, actions={actions.shape}")

    B, K, A = preds.shape

    if is_pad is None:
        valid = torch.ones((B, K), dtype=torch.bool, device=preds.device)
    else:
        if is_pad.shape != (B, K):
            raise ValueError(f"is_pad must be [B,K], got {is_pad.shape} vs expected {(B,K)}")
        valid = ~is_pad.to(device=preds.device)

    # per-element loss: [B,K,A]
    if use_smooth_l1:
        per_elem = F.smooth_l1_loss(preds, actions, reduction="none", beta=smooth_l1_beta)
    else:
        per_elem = F.l1_loss(preds, actions, reduction="none")

    # mask: [B,K,1]
    mask = valid.unsqueeze(-1).to(per_elem.dtype)

    # masked sum
    masked_sum = (per_elem * mask).sum()

    # normalization: average over valid elements
    denom = mask.sum() * A  # mask.sum() is valid steps count, times A dims
    denom = torch.clamp(denom, min=1.0)

    if reduction == "mean":
        loss = masked_sum / denom
    elif reduction == "sum":
        loss = masked_sum
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    loss_dict = {
        "l1": float(loss.detach().cpu().item()),
        "valid_steps": float(valid.sum().detach().cpu().item()),
    }
    return loss, loss_dict


def act_chunk_mse_loss(
    preds: torch.Tensor,
    actions: torch.Tensor,
    is_pad: Optional[torch.Tensor] = None,
    *,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Same as act_chunk_l1_loss but with MSE.
    """
    if preds.dim() != 3 or actions.dim() != 3:
        raise ValueError(f"preds/actions must be 3D [B,K,A], got preds={preds.shape}, actions={actions.shape}")
    if preds.shape != actions.shape:
        raise ValueError(f"preds and actions shapes must match, got preds={preds.shape}, actions={actions.shape}")

    B, K, A = preds.shape

    if is_pad is None:
        valid = torch.ones((B, K), dtype=torch.bool, device=preds.device)
    else:
        if is_pad.shape != (B, K):
            raise ValueError(f"is_pad must be [B,K], got {is_pad.shape} vs expected {(B,K)}")
        valid = ~is_pad.to(device=preds.device)

    per_elem = F.mse_loss(preds, actions, reduction="none")  # [B,K,A]
    mask = valid.unsqueeze(-1).to(per_elem.dtype)

    masked_sum = (per_elem * mask).sum()
    denom = mask.sum() * A
    denom = torch.clamp(denom, min=1.0)

    if reduction == "mean":
        loss = masked_sum / denom
    elif reduction == "sum":
        loss = masked_sum
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    loss_dict = {
        "mse": float(loss.detach().cpu().item()),
        "valid_steps": float(valid.sum().detach().cpu().item()),
    }
    return loss, loss_dict