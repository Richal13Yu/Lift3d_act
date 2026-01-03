#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import pathlib
from typing import Any, Dict, Optional, Tuple

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
    # object (ActOutput)
    for attr in ("actions", "a_hat", "action", "pred_actions", "actions_hat"):
        if hasattr(preds, attr):
            v = getattr(preds, attr)
            if torch.is_tensor(v):
                return v
    raise ValueError(f"Unsupported model output type={type(preds)}")


# ----------------------------
# Checkpoint helpers
# ----------------------------
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

    summary = {
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_preview": missing[:30],
        "unexpected_keys_preview": unexpected[:30],
    }
    if missing:
        _log_warn(f"Missing keys: {len(missing)} (show up to 30)\n{missing[:30]}")
    if unexpected:
        _log_warn(f"Unexpected keys: {len(unexpected)} (show up to 30)\n{unexpected[:30]}")
    return summary


# ----------------------------
# Diagnostics + plots
# ----------------------------
def _drop_pad(a_pred: np.ndarray, a_gt: np.ndarray, pad: Optional[np.ndarray]):
    # a_*: [K,A]
    if pad is None:
        return a_pred, a_gt, a_pred.shape[0]
    valid = ~pad.astype(bool)
    a_pred2 = a_pred[valid]
    a_gt2 = a_gt[valid]
    return a_pred2, a_gt2, int(valid.sum())


def diagnose_chunk(pred: np.ndarray, gt: np.ndarray, eps_abs: float = 1e-6, repeat_ratio_thresh: float = 1e-3) -> Dict[str, Any]:
    """
    pred/gt: [K,A]
    """
    K, A = pred.shape
    # absolute diag (numerical)
    step_change = np.abs(np.diff(pred, axis=0))          # [K-1,A]
    max_step_change_abs = float(step_change.max()) if K > 1 else 0.0
    pred_std = np.std(pred, axis=0)                      # [A]
    gt_std = np.std(gt, axis=0)
    max_std_over_time_dim = float(pred_std.max())
    looks_repeat_abs = (max_step_change_abs < eps_abs) or (max_std_over_time_dim < eps_abs)

    # relative diag (compared to GT)
    pred_l2_from0 = np.linalg.norm(pred - pred[0:1], axis=1)   # [K]
    gt_l2_from0 = np.linalg.norm(gt - gt[0:1], axis=1)
    pred_l2_from0_max = float(pred_l2_from0.max())
    gt_l2_from0_max = float(gt_l2_from0.max())
    variation_ratio_pred_over_gt = float(pred_l2_from0_max / (gt_l2_from0_max + 1e-12))

    std_ratio = pred_std / (gt_std + 1e-12)
    std_ratio_max = float(std_ratio.max())
    std_ratio_mean = float(std_ratio.mean())

    looks_like_repeat_relative = variation_ratio_pred_over_gt < repeat_ratio_thresh

    abs_diag = {
        "K": int(K),
        "A": int(A),
        "max_step_change_abs": max_step_change_abs,
        "max_std_over_time_dim": max_std_over_time_dim,
        "looks_repeat_abs": bool(looks_repeat_abs),
        "eps_abs": float(eps_abs),
    }
    rel_diag = {
        "pred_l2_from0_max": pred_l2_from0_max,
        "gt_l2_from0_max": gt_l2_from0_max,
        "variation_ratio_pred_over_gt": variation_ratio_pred_over_gt,
        "pred_std_max": float(pred_std.max()),
        "gt_std_max": float(gt_std.max()),
        "std_ratio_max": std_ratio_max,
        "std_ratio_mean": std_ratio_mean,
        "repeat_ratio_thresh": float(repeat_ratio_thresh),
        "looks_like_repeat_relative_to_gt": bool(looks_like_repeat_relative),
    }
    return {"abs_diag": abs_diag, "rel_diag": rel_diag}


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
    # std over time, per-dim
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
    _log_info(f"[INFO] Inference eval with {colored(pathlib.Path(__file__).absolute(), 'red')}")
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

    plot_index = getattr(eval_cfg, "plot_index", 0) if eval_cfg is not None else 0
    eps_abs = float(getattr(eval_cfg, "chunk_eps_abs", 1e-6)) if eval_cfg is not None else 1e-6
    repeat_ratio_thresh = float(getattr(eval_cfg, "repeat_ratio_thresh", 1e-3)) if eval_cfg is not None else 1e-3

    ckpt_file = _resolve_checkpoint_path(ckpt_path)
    _log_info(f"[INFO] Using checkpoint: {colored(ckpt_file, 'green')}")

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

    if plot_index < 0 or plot_index >= len(dataset):
        raise ValueError(f"plot_index out of range: {plot_index} (len={len(dataset)})")

    batch = None
    for i, b in enumerate(loader):
        if i == plot_index:
            batch = b
            break
    assert batch is not None

    images, point_clouds, robot_states, raw_states, actions, texts, is_pad = _unpack_batch(batch)

    robot_state_dim = int(robot_states.size(-1))
    action_dim = int(actions.size(-1))
    gt_K = int(actions.size(1)) if actions.dim() == 3 else 1
    _log_info(f"[INFO] Robot state dim: {robot_state_dim}")
    _log_info(f"[INFO] Action dim: {action_dim}")
    _log_info(f"[INFO] GT chunk K: {gt_K}")

    # model
    model = instantiate(
        config=config.agent.instantiate_config,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
    ).to(config.device)

    ckpt_summary = _load_checkpoint(model, ckpt_file)
    _log_info(f"[INFO] ckpt load summary: missing={ckpt_summary['missing_keys_count']}, unexpected={ckpt_summary['unexpected_keys_count']}")

    model.eval()

    images = images.to(config.device)
    point_clouds = point_clouds.to(config.device)
    robot_states = robot_states.to(config.device)
    actions = actions.to(config.device)
    if is_pad is not None:
        is_pad = is_pad.to(config.device)

    # ----------------------------
    # Forward A: posterior (TRAIN-LIKE) 传 actions
    # ----------------------------
    with torch.no_grad():
        preds_post = model(images, point_clouds, robot_states, texts, actions=actions, is_pad=is_pad)
    a_post = _get_actions_pred(preds_post).detach().cpu().numpy()  # [1,K,A]
    a_gt = actions.detach().cpu().numpy()

    a_post = a_post[0] if a_post.ndim == 3 else a_post
    a_gt_np = a_gt[0] if a_gt.ndim == 3 else a_gt

    pad_np = is_pad.detach().cpu().numpy()[0] if is_pad is not None else None
    a_post, a_gt_np_post, K_valid = _drop_pad(a_post, a_gt_np, pad_np)

    plot_actions(
        a_post, a_gt_np_post,
        save_path=str(out_dir / f"plot_idx{plot_index:05d}_actions_posterior.png"),
        title=f"Posterior (train-like, with GT actions) | split={split} | idx={plot_index}"
    )

    mse_post = float(np.mean((a_post - a_gt_np_post) ** 2))
    mae_post = float(np.mean(np.abs(a_post - a_gt_np_post)))
    _log_info(f"[POSTERIOR] K_valid={K_valid} mse={mse_post:.6e} mae={mae_post:.6e}")

    # ----------------------------
    # Forward B: inference (REAL) 不传 actions
    # ----------------------------
    with torch.no_grad():
        preds_inf = model(images, point_clouds, robot_states, texts)
    a_inf = _get_actions_pred(preds_inf).detach().cpu().numpy()

    _log_info(f"[INFO] RAW model pred tensor shape: {tuple(a_inf.shape)}")

    a_inf = a_inf[0] if a_inf.ndim == 3 else a_inf
    # align K with gt_K if needed
    if a_inf.shape[0] != a_gt_np.shape[0]:
        K = min(a_inf.shape[0], a_gt_np.shape[0])
        a_inf = a_inf[:K]
        a_gt_np_inf = a_gt_np[:K]
        pad_np2 = pad_np[:K] if pad_np is not None else None
    else:
        a_gt_np_inf = a_gt_np
        pad_np2 = pad_np

    a_inf, a_gt_np_inf, K_valid2 = _drop_pad(a_inf, a_gt_np_inf, pad_np2)
    _log_info(f"[INFO] After padding mask: K_valid={K_valid2}")

    diag = diagnose_chunk(a_inf, a_gt_np_inf, eps_abs=eps_abs, repeat_ratio_thresh=repeat_ratio_thresh)
    abs_diag, rel_diag = diag["abs_diag"], diag["rel_diag"]

    _log_info("[DIAG-ABS] " + json.dumps(abs_diag, indent=2))
    _log_info("[DIAG-REL] " + json.dumps(rel_diag, indent=2))

    _log_info(f"[PROOF] pred[0][:5]={a_inf[0][:5]}")
    if a_inf.shape[0] > 1:
        _log_info(f"[PROOF] pred[1][:5]={a_inf[1][:5]}")
    if a_inf.shape[0] > 2:
        _log_info(f"[PROOF] pred[2][:5]={a_inf[2][:5]}")

    # “一行判据”（也写在日志里）
    var_ratio = rel_diag["variation_ratio_pred_over_gt"]
    _log_info(f"[ONE-LINE] var_ratio(pred/gt)={var_ratio:.3e}  (<<1e-3 means effectively repeating)")

    if rel_diag["looks_like_repeat_relative_to_gt"]:
        _log_warn("Pred changes over time are tiny compared to GT -> effectively repeating the same action across the chunk.")
    else:
        _log_info("Pred shows meaningful time variation relative to GT (not a flat chunk).")

    # plots
    plot_actions(
        a_inf, a_gt_np_inf,
        save_path=str(out_dir / f"plot_idx{plot_index:05d}_actions_inference.png"),
        title=f"Inference (no GT actions) | split={split} | idx={plot_index}"
    )
    plot_variation(
        a_inf, a_gt_np_inf,
        save_path=str(out_dir / f"plot_idx{plot_index:05d}_chunk_variation_pred_vs_gt_inference.png"),
        title="Std over chunk (pred vs gt) | inference"
    )

    mse_inf = float(np.mean((a_inf - a_gt_np_inf) ** 2))
    mae_inf = float(np.mean(np.abs(a_inf - a_gt_np_inf)))

    summary = {
        "mode": "inference_eval",
        "gt_K": int(gt_K),
        "used_K_after_align_and_pad": int(K_valid2),
        "raw_pred_shape": list(map(int, list(a_inf.shape if isinstance(a_inf, np.ndarray) else []))),
        "abs_diag": abs_diag,
        "rel_diag": rel_diag,
        "posterior_mse": mse_post,
        "posterior_mae": mae_post,
        "inference_mse": mse_inf,
        "inference_mae": mae_inf,
        "ckpt_summary": {
            "missing_keys_count": ckpt_summary["missing_keys_count"],
            "unexpected_keys_count": ckpt_summary["unexpected_keys_count"],
        },
        "params": {
            "chunk_eps_abs": eps_abs,
            "repeat_ratio_thresh": repeat_ratio_thresh,
        },
    }
    out_json = out_dir / f"plot_idx{plot_index:05d}_chunk_diagnosis_inference.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    _log_info(f"[OK] Saved diagnosis json: {colored(str(out_json), 'green')}")
    _log_info(f"[INFERENCE] mse={mse_inf:.6e} mae={mae_inf:.6e}")


if __name__ == "__main__":
    main()