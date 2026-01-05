#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lift3d/tools/eval_act.py

Offline eval for ACT-style chunk policy.

It evaluates TWO modes:
1) POSTERIOR (train-like): model(..., actions=gt_actions, is_pad=gt_is_pad)
2) INFERENCE (real):       model(..., actions=None)

Supports model output types:
- Tensor
- dict with "actions"/"a_hat"
- ActOutput object with .actions

Metrics:
- per-step MAE/MSE averaged over ALL valid (non-pad) steps in dataset split
- optional plots + chunk flatness diagnostics on selected indices

Run examples:
  python -m lift3d.tools.eval_act \
    task_name=peg_recover dataset_dir=... benchmark=act_offline agent=lift3d_act \
    evaluation.checkpoint_path=/path/to/best_model.pth evaluation.split=validation \
    evaluation.plot_indices='[0,10]' evaluation.max_batches=200

Or from your bash:
  HYDRA_FULL_ERROR=1 python lift3d/tools/eval_act.py evaluation.checkpoint_path=...
"""

from __future__ import annotations

import os
import json
import pathlib
from typing import Any, Dict, Optional, Tuple, List

import hydra
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hydra.utils import instantiate
from termcolor import colored

from lift3d.helpers.common import Logger, set_seed


# ----------------------------
# Logger compat
# ----------------------------
def _log_info(msg: str):
    if hasattr(Logger, "log_info"):
        Logger.log_info(msg)
    else:
        print(msg)

def _log_warn(msg: str):
    if hasattr(Logger, "log_warn"):
        Logger.log_warn(msg)
    elif hasattr(Logger, "log_warning"):
        Logger.log_warning(msg)
    else:
        print("[WARNING]", msg)

def _print_sep():
    if hasattr(Logger, "print_seperator"):
        Logger.print_seperator()
    else:
        print("-" * 100)


# ----------------------------
# Batch helpers
# ----------------------------
def _unpack_batch(
    batch,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor, Any, Optional[torch.Tensor]]:
    """
    Expected dataset batch formats:
      - 6-tuple: images, point_clouds, robot_states, raw_states, actions, texts
      - 7-tuple: images, point_clouds, robot_states, raw_states, actions, texts, is_pad
    """
    if not isinstance(batch, (tuple, list)):
        raise ValueError(f"Batch must be tuple/list, got {type(batch)}")

    if len(batch) == 6:
        images, point_clouds, robot_states, raw_states, actions, texts = batch
        is_pad = None
    elif len(batch) == 7:
        images, point_clouds, robot_states, raw_states, actions, texts, is_pad = batch
    else:
        raise ValueError(f"Unexpected batch size {len(batch)}; expected 6 or 7.")

    return images, point_clouds, robot_states, raw_states, actions, texts, is_pad


def _get_actions_pred(preds) -> torch.Tensor:
    """Extract action prediction tensor from model output."""
    if torch.is_tensor(preds):
        return preds
    if isinstance(preds, dict):
        for k in ("actions", "a_hat"):
            if k in preds and torch.is_tensor(preds[k]):
                return preds[k]
        raise ValueError(f"Dict output has no tensor 'actions'/'a_hat'. keys={list(preds.keys())}")

    # ActOutput / custom object
    for attr in ("actions", "a_hat", "action", "pred_actions", "actions_hat"):
        if hasattr(preds, attr):
            v = getattr(preds, attr)
            if torch.is_tensor(v):
                return v

    raise ValueError(f"Unsupported model output type={type(preds)}")


def _ensure_chunk(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize to [B,K,A].
    Accepts:
      [B,A] -> [B,1,A]
      [B,K,A] -> unchanged
    """
    if x.dim() == 2:
        return x[:, None, :]
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected action tensor [B,A] or [B,K,A], got {tuple(x.shape)}")


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefixes=("module.", "model.")) -> Dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    for p in prefixes:
        if len(keys) > 0 and all(k.startswith(p) for k in keys):
            return {k[len(p):]: v for k, v in state_dict.items()}
    return state_dict


def _resolve_checkpoint_path(ckpt_path: str) -> str:
    p = pathlib.Path(os.path.expanduser(ckpt_path))
    if p.exists() and p.is_file():
        return str(p)
    if p.exists() and p.is_dir():
        for name in ["best_model.pth", "last_model.pth", "model.pth", "checkpoint.pth", "ckpt.pth"]:
            c = p / name
            if c.exists() and c.is_file():
                return str(c)
        pths = sorted(p.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
        if len(pths) > 0:
            return str(pths[0])
    raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")


def _load_checkpoint(model: torch.nn.Module, ckpt_file: str) -> Dict[str, Any]:
    obj = torch.load(ckpt_file, map_location="cpu")

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state_dict = obj["model"]
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        state_dict = obj
    else:
        raise ValueError(f"Unsupported checkpoint format at {ckpt_file} (type={type(obj)})")

    state_dict = _strip_prefix_if_present(state_dict, prefixes=("module.", "model."))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        _log_warn(f"Missing keys: {len(missing)} (show up to 20)\n{missing[:20]}")
    if unexpected:
        _log_warn(f"Unexpected keys: {len(unexpected)} (show up to 20)\n{unexpected[:20]}")

    return {
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_preview": missing[:20],
        "unexpected_keys_preview": unexpected[:20],
    }


# ----------------------------
# Diagnostics + plots
# ----------------------------
def _valid_mask_from_is_pad(is_pad: Optional[torch.Tensor], K: int, device) -> torch.Tensor:
    """
    Return boolean mask [B,K] where True=valid.
    If is_pad is None -> all valid.
    """
    if is_pad is None:
        return torch.ones((1, K), dtype=torch.bool, device=device)
    is_pad = is_pad.to(device)
    if is_pad.dim() == 2:
        return (~is_pad.bool())
    if is_pad.dim() == 1:
        return (~is_pad.bool())[None, :]
    raise ValueError(f"is_pad must be [B,K] or [K], got {tuple(is_pad.shape)}")


def diagnose_chunk_np(pred: np.ndarray, gt: np.ndarray, eps_abs: float = 1e-6, repeat_ratio_thresh: float = 1e-3) -> Dict[str, Any]:
    """
    pred/gt: [K,A]
    """
    K, A = pred.shape

    step_change = np.abs(np.diff(pred, axis=0))  # [K-1,A]
    max_step_change_abs = float(step_change.max()) if K > 1 else 0.0
    pred_std = np.std(pred, axis=0)
    gt_std = np.std(gt, axis=0)
    max_std_over_time_dim = float(pred_std.max())
    looks_repeat_abs = (max_step_change_abs < eps_abs) or (max_std_over_time_dim < eps_abs)

    pred_l2_from0 = np.linalg.norm(pred - pred[0:1], axis=1)
    gt_l2_from0 = np.linalg.norm(gt - gt[0:1], axis=1)
    pred_l2_from0_max = float(pred_l2_from0.max())
    gt_l2_from0_max = float(gt_l2_from0.max())
    variation_ratio = float(pred_l2_from0_max / (gt_l2_from0_max + 1e-12))
    looks_repeat_rel = variation_ratio < repeat_ratio_thresh

    std_ratio = pred_std / (gt_std + 1e-12)

    return {
        "abs_diag": {
            "K": int(K),
            "A": int(A),
            "max_step_change_abs": max_step_change_abs,
            "max_std_over_time_dim": max_std_over_time_dim,
            "looks_repeat_abs": bool(looks_repeat_abs),
            "eps_abs": float(eps_abs),
        },
        "rel_diag": {
            "pred_l2_from0_max": pred_l2_from0_max,
            "gt_l2_from0_max": gt_l2_from0_max,
            "variation_ratio_pred_over_gt": variation_ratio,
            "pred_std_max": float(pred_std.max()),
            "gt_std_max": float(gt_std.max()),
            "std_ratio_max": float(std_ratio.max()),
            "std_ratio_mean": float(std_ratio.mean()),
            "repeat_ratio_thresh": float(repeat_ratio_thresh),
            "looks_like_repeat_relative_to_gt": bool(looks_repeat_rel),
        },
    }


def plot_actions(pred: np.ndarray, gt: np.ndarray, save_path: str, title: str):
    K, A = pred.shape
    t = np.arange(K)
    fig, axes = plt.subplots(A, 1, figsize=(14, 2.2 * A), sharex=True)
    if A == 1:
        axes = [axes]
    fig.suptitle(title)
    for i in range(A):
        ax = axes[i]
        ax.plot(t, pred[:, i], label="pred", alpha=0.9)
        ax.plot(t, gt[:, i], label="gt", alpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(f"dim {i}")
        ax.legend()
    axes[-1].set_xlabel("k")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_variation(pred: np.ndarray, gt: np.ndarray, save_path: str, title: str):
    pred_std = np.std(pred, axis=0)
    gt_std = np.std(gt, axis=0)
    x = np.arange(pred.shape[1])

    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, pred_std, marker="o", label="pred_std_over_k")
    ax.plot(x, gt_std, marker="o", label="gt_std_over_k")
    ax.set_title(title)
    ax.set_xlabel("action_dim")
    ax.set_ylabel("std over chunk (k)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(config):
    _log_info(f"[INFO] Eval ACT with {colored(pathlib.Path(__file__).absolute(), 'red')}")
    _log_info(f"[INFO] Task: {colored(config.task_name, 'green')}")
    _log_info(f"[INFO] Dataset dir: {colored(config.dataset_dir, 'green')}")
    _log_info(f"[INFO] Device: {colored(config.device, 'green')}")
    _print_sep()

    set_seed(config.seed)

    out_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = getattr(config, "evaluation", None)

    split = getattr(eval_cfg, "split", "validation") if eval_cfg is not None else "validation"
    ckpt_path = getattr(eval_cfg, "checkpoint_path", None) if eval_cfg is not None else None
    if ckpt_path is None and eval_cfg is not None:
        ckpt_path = getattr(eval_cfg, "ckpt_path", None)
    if ckpt_path is None:
        raise ValueError("Please provide checkpoint path via +evaluation.checkpoint_path=...")

    # optional controls
    max_batches = getattr(eval_cfg, "max_batches", None) if eval_cfg is not None else None
    plot_indices = getattr(eval_cfg, "plot_indices", []) if eval_cfg is not None else []
    if isinstance(plot_indices, int):
        plot_indices = [plot_indices]
    if plot_indices is None:
        plot_indices = []
    plot_indices = list(plot_indices)

    eps_abs = float(getattr(eval_cfg, "chunk_eps_abs", 1e-6)) if eval_cfg is not None else 1e-6
    repeat_ratio_thresh = float(getattr(eval_cfg, "repeat_ratio_thresh", 1e-3)) if eval_cfg is not None else 1e-3

    # IMPORTANT: make inference deterministic as much as possible
    # (your actor currently samples z ~ N(0,1) in inference; this makes eval reproducible)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    ckpt_file = _resolve_checkpoint_path(ckpt_path)
    _log_info(f"[INFO] Using checkpoint: {colored(ckpt_file, 'green')}")
    _log_info(f"[INFO] Split: {colored(split, 'green')}")

    # dataset / loader
    dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        data_dir=config.dataset_dir,
        split=split,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split '{split}' is empty.")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=getattr(config.dataloader, "num_workers", 0),
        shuffle=False,
        pin_memory=getattr(config.dataloader, "pin_memory", False),
        drop_last=False,
    )

    # infer dims from first batch
    first_batch = next(iter(loader))
    images, point_clouds, robot_states, raw_states, actions, texts, is_pad = _unpack_batch(first_batch)
    robot_state_dim = int(robot_states.size(-1))
    action_dim = int(actions.size(-1))

    _log_info(f"[INFO] Robot state dim: {robot_state_dim}")
    _log_info(f"[INFO] Action dim: {action_dim}")

    # model
    model = instantiate(
        config=config.agent.instantiate_config,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
    ).to(config.device)

    ckpt_summary = _load_checkpoint(model, ckpt_file)
    _log_info(f"[INFO] ckpt load summary: missing={ckpt_summary['missing_keys_count']}, unexpected={ckpt_summary['unexpected_keys_count']}")
    model.eval()

    # accumulate metrics over dataset
    sum_mae_post = 0.0
    sum_mse_post = 0.0
    sum_mae_inf = 0.0
    sum_mse_inf = 0.0
    n_valid_steps = 0

    # optional diag snapshots
    diag_records: List[Dict[str, Any]] = []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            break

        images, point_clouds, robot_states, raw_states, actions, texts, is_pad = _unpack_batch(batch)

        images = images.to(config.device)
        point_clouds = point_clouds.to(config.device)
        robot_states = robot_states.to(config.device)
        actions = actions.to(config.device)
        if is_pad is not None:
            is_pad = is_pad.to(config.device)

        # GT chunk: [1,K,A]
        a_gt = _ensure_chunk(actions)
        K = a_gt.shape[1]
        valid_mask = _valid_mask_from_is_pad(is_pad, K, device=a_gt.device)  # [1,K]

        # ----------------------------
        # POSTERIOR: pass GT actions
        # ----------------------------
        with torch.no_grad():
            preds_post = model(images, point_clouds, robot_states, texts, actions=a_gt, is_pad=is_pad)
        a_post = _ensure_chunk(_get_actions_pred(preds_post))  # [1,K,A]

        # ----------------------------
        # INFERENCE: no GT actions
        # ----------------------------
        with torch.no_grad():
            preds_inf = model(images, point_clouds, robot_states, texts)
        a_inf = _ensure_chunk(_get_actions_pred(preds_inf))  # [1,K',A]
        # align K if needed
        if a_inf.shape[1] != K:
            K2 = min(a_inf.shape[1], K)
            a_inf = a_inf[:, :K2, :]
            a_post = a_post[:, :K2, :]
            a_gt = a_gt[:, :K2, :]
            valid_mask = valid_mask[:, :K2]
            K = K2

        # compute per-step errors on valid steps only
        # flatten [1,K,A] -> [K,A]
        vm = valid_mask[0]  # [K]
        if vm.sum().item() == 0:
            continue

        post_valid = a_post[0, vm, :]
        inf_valid = a_inf[0, vm, :]
        gt_valid = a_gt[0, vm, :]

        mae_post = torch.mean(torch.abs(post_valid - gt_valid)).item()
        mse_post = torch.mean((post_valid - gt_valid) ** 2).item()
        mae_inf = torch.mean(torch.abs(inf_valid - gt_valid)).item()
        mse_inf = torch.mean((inf_valid - gt_valid) ** 2).item()

        # weight by number of valid steps (so pad 不会影响平均)
        w = int(vm.sum().item())
        sum_mae_post += mae_post * w
        sum_mse_post += mse_post * w
        sum_mae_inf += mae_inf * w
        sum_mse_inf += mse_inf * w
        n_valid_steps += w

        # optional plot/diag for selected indices (dataset index == batch index because shuffle=False)
        if bi in plot_indices:
            p_post = post_valid.detach().cpu().numpy()
            p_inf = inf_valid.detach().cpu().numpy()
            g_np = gt_valid.detach().cpu().numpy()

            # diag on inference chunk
            diag = diagnose_chunk_np(p_inf, g_np, eps_abs=eps_abs, repeat_ratio_thresh=repeat_ratio_thresh)
            diag_records.append({"index": bi, "K_valid": w, **diag})

            plot_actions(
                p_inf, g_np,
                save_path=str(out_dir / f"idx{bi:05d}_actions_inference.png"),
                title=f"Inference (no GT actions) | split={split} | idx={bi}"
            )
            plot_variation(
                p_inf, g_np,
                save_path=str(out_dir / f"idx{bi:05d}_variation_inference.png"),
                title=f"Std over chunk (pred vs gt) | inference | idx={bi}"
            )
            plot_actions(
                p_post, g_np,
                save_path=str(out_dir / f"idx{bi:05d}_actions_posterior.png"),
                title=f"Posterior (with GT actions) | split={split} | idx={bi}"
            )

            rel = diag["rel_diag"]
            _log_info(f"[DIAG idx={bi}] var_ratio(pred/gt)={rel['variation_ratio_pred_over_gt']:.3e} "
                      f"repeat={rel['looks_like_repeat_relative_to_gt']}")

    if n_valid_steps == 0:
        _log_warn("No valid steps found (all padded or empty loader).")
        return

    # global averages
    avg_mae_post = sum_mae_post / n_valid_steps
    avg_mse_post = sum_mse_post / n_valid_steps
    avg_mae_inf = sum_mae_inf / n_valid_steps
    avg_mse_inf = sum_mse_inf / n_valid_steps

    _print_sep()
    _log_info(f"[POSTERIOR] avg_mae={avg_mae_post:.6e} avg_mse={avg_mse_post:.6e} (over {n_valid_steps} valid steps)")
    _log_info(f"[INFERENCE ] avg_mae={avg_mae_inf:.6e} avg_mse={avg_mse_inf:.6e} (over {n_valid_steps} valid steps)")

    summary = {
        "split": split,
        "checkpoint": ckpt_file,
        "n_valid_steps": int(n_valid_steps),
        "posterior": {"avg_mae": float(avg_mae_post), "avg_mse": float(avg_mse_post)},
        "inference": {"avg_mae": float(avg_mae_inf), "avg_mse": float(avg_mse_inf)},
        "ckpt_summary": {
            "missing_keys_count": ckpt_summary["missing_keys_count"],
            "unexpected_keys_count": ckpt_summary["unexpected_keys_count"],
        },
        "diag_records": diag_records,
        "params": {
            "chunk_eps_abs": float(eps_abs),
            "repeat_ratio_thresh": float(repeat_ratio_thresh),
            "plot_indices": plot_indices,
            "max_batches": None if max_batches is None else int(max_batches),
        },
    }

    out_json = out_dir / "eval_act_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    _log_info(f"[OK] Saved summary json: {colored(str(out_json), 'green')}")
    _log_info(f"[OK] Output dir: {colored(str(out_dir), 'green')}")


if __name__ == "__main__":
    main()