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
        _log_info(f"[WARNING] {msg}")

def _print_sep():
    if hasattr(Logger, "print_seperator"):
        Logger.print_seperator()
    else:
        print("-" * 120)


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

    for attr in ("actions", "a_hat", "action", "pred_actions", "actions_hat"):
        if hasattr(preds, attr):
            v = getattr(preds, attr)
            if torch.is_tensor(v):
                return v

    keys = None
    if hasattr(preds, "__dict__"):
        keys = list(preds.__dict__.keys())
    raise ValueError(
        "Model output must be a Tensor, a dict containing 'actions'/'a_hat', "
        "or an object with tensor attribute 'actions'/'a_hat'. "
        f"Got type={type(preds)} attrs={keys}"
    )


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


def _load_checkpoint(model: torch.nn.Module, ckpt_file: str, device: str) -> Dict[str, Any]:
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
        _log_warn(f"Missing keys: {len(missing)} (show up to 30)\n{missing[:30]}")
    if unexpected:
        _log_warn(f"Unexpected keys: {len(unexpected)} (show up to 30)\n{unexpected[:30]}")

    model.to(device)

    return {
        "missing_keys_count": int(len(missing)),
        "unexpected_keys_count": int(len(unexpected)),
        "missing_keys_preview": missing[:30] if missing else [],
        "unexpected_keys_preview": unexpected[:30] if unexpected else [],
    }


# ----------------------------
# Diagnostics helpers
# ----------------------------
def _unique_rows_rounded(x: np.ndarray, decimals: int = 8) -> int:
    rr = np.round(x.astype(np.float64), decimals=decimals)
    return int(np.unique(rr, axis=0).shape[0])

def _variation_curves(a: np.ndarray) -> Dict[str, np.ndarray]:
    """
    a: [K,A]
    returns:
      l2_from0[t] = ||a[t]-a[0]||
      l2_step[t]  = ||a[t]-a[t-1]|| for t>=1
    """
    K = a.shape[0]
    diff0 = a - a[0:1]
    l2_from0 = np.linalg.norm(diff0, axis=1)
    if K > 1:
        step = a[1:] - a[:-1]
        l2_step = np.linalg.norm(step, axis=1)
    else:
        l2_step = np.zeros((0,), dtype=np.float64)
    return {"l2_from0": l2_from0, "l2_step": l2_step}

def diagnose_chunk_absolute(a_pred: np.ndarray, eps_abs: float) -> Dict[str, Any]:
    """
    Absolute diagnosis: tells if pred changes over time at all (numerically).
    """
    K, A = a_pred.shape
    if K <= 1:
        return {"K": int(K), "A": int(A), "looks_repeat_abs": True, "eps_abs": float(eps_abs)}

    step_diff = a_pred[1:] - a_pred[:-1]
    max_step_change_abs = float(np.max(np.abs(step_diff)))
    std_over_time = np.std(a_pred, axis=0)
    max_std_dim = float(np.max(std_over_time))
    uniq = _unique_rows_rounded(a_pred, decimals=8)

    looks_repeat_abs = (max_step_change_abs <= eps_abs) or (max_std_dim <= eps_abs) or (uniq <= 2)

    return {
        "K": int(K),
        "A": int(A),
        "max_step_change_abs": max_step_change_abs,
        "max_std_over_time_dim": max_std_dim,
        "unique_rows_rounded_8dp": int(uniq),
        "looks_repeat_abs": bool(looks_repeat_abs),
        "eps_abs": float(eps_abs),
    }

def diagnose_chunk_relative_to_gt(a_pred: np.ndarray, a_gt: np.ndarray, repeat_ratio_thresh: float = 1e-3) -> Dict[str, Any]:
    """
    Relative diagnosis (recommended):
      If pred time-variation is tiny compared to gt time-variation, treat as "repeat" in practice.
    """
    assert a_pred.shape == a_gt.shape, f"pred {a_pred.shape} vs gt {a_gt.shape}"
    K, A = a_pred.shape

    pred_std = np.std(a_pred, axis=0)
    gt_std = np.std(a_gt, axis=0)
    std_ratio = pred_std / (gt_std + 1e-12)

    pred_range = np.max(a_pred, axis=0) - np.min(a_pred, axis=0)
    gt_range = np.max(a_gt, axis=0) - np.min(a_gt, axis=0)
    range_ratio = pred_range / (gt_range + 1e-12)

    vp = _variation_curves(a_pred)
    vg = _variation_curves(a_gt)

    pred_l2_from0_max = float(np.max(vp["l2_from0"])) if K > 0 else 0.0
    gt_l2_from0_max = float(np.max(vg["l2_from0"])) if K > 0 else 0.0
    variation_ratio = pred_l2_from0_max / (gt_l2_from0_max + 1e-12)

    looks_repeat_rel = (variation_ratio < repeat_ratio_thresh) and (float(np.max(std_ratio)) < repeat_ratio_thresh * 10)

    return {
        "K": int(K),
        "A": int(A),
        "pred_std_max": float(np.max(pred_std)),
        "pred_std_mean": float(np.mean(pred_std)),
        "gt_std_max": float(np.max(gt_std)),
        "gt_std_mean": float(np.mean(gt_std)),
        "std_ratio_max": float(np.max(std_ratio)),
        "std_ratio_mean": float(np.mean(std_ratio)),
        "pred_range_max": float(np.max(pred_range)),
        "pred_range_mean": float(np.mean(pred_range)),
        "gt_range_max": float(np.max(gt_range)),
        "gt_range_mean": float(np.mean(gt_range)),
        "range_ratio_max": float(np.max(range_ratio)),
        "range_ratio_mean": float(np.mean(range_ratio)),
        "pred_l2_from0_max": pred_l2_from0_max,
        "gt_l2_from0_max": gt_l2_from0_max,
        "variation_ratio_pred_over_gt": variation_ratio,
        "repeat_ratio_thresh": float(repeat_ratio_thresh),
        "looks_like_repeat_relative_to_gt": bool(looks_repeat_rel),
    }

def plot_pred_gt_variation(a_pred: np.ndarray, a_gt: np.ndarray, save_path: str, title: str):
    """
    Single figure: pred and gt variation curves together.
    If pred is "flat" you will see pred curves near 0 while gt curves are larger.
    """
    vp = _variation_curves(a_pred)
    vg = _variation_curves(a_gt)
    K = a_pred.shape[0]

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)

    ax.plot(np.arange(K), vp["l2_from0"], label="pred: L2(pred[t]-pred[0])", alpha=0.9)
    ax.plot(np.arange(K), vg["l2_from0"], label="gt:   L2(gt[t]-gt[0])", alpha=0.9)

    if K > 1:
        ax.plot(np.arange(1, K), vp["l2_step"], label="pred: L2(pred[t]-pred[t-1])", alpha=0.9)
        ax.plot(np.arange(1, K), vg["l2_step"], label="gt:   L2(gt[t]-gt[t-1])", alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel("L2 norm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_actions_1d(pred: np.ndarray, gt: np.ndarray, save_path: str, title: str):
    T = pred.shape[0]
    A = pred.shape[1]
    time = np.arange(T)

    fig, axes = plt.subplots(A, 1, figsize=(14, 2.3 * A), sharex=True)
    if A == 1:
        axes = [axes]
    fig.suptitle(title)

    for i in range(A):
        ax = axes[i]
        ax.plot(time, pred[:, i], label="pred", alpha=0.8)
        ax.plot(time, gt[:, i], label="gt", alpha=0.8)
        mse = float(np.mean((pred[:, i] - gt[:, i]) ** 2))
        mae = float(np.mean(np.abs(pred[:, i] - gt[:, i])))
        ax.set_ylabel(f"dim {i}")
        ax.set_title(f"MSE={mse:.6f}, MAE={mae:.6f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("timestep")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(config):
    _log_info(f"[INFO] Plot eval with {colored(pathlib.Path(__file__).absolute(), 'red')}")
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

    plot_index = int(getattr(eval_cfg, "plot_index", 0)) if eval_cfg is not None else 0
    debug_shapes = bool(getattr(eval_cfg, "debug_shapes", True)) if eval_cfg is not None else True

    # diagnosis knobs
    eps_abs = float(getattr(eval_cfg, "chunk_eps_abs", 1e-6)) if eval_cfg is not None else 1e-6
    repeat_ratio_thresh = float(getattr(eval_cfg, "repeat_ratio_thresh", 1e-3)) if eval_cfg is not None else 1e-3

    if ckpt_path is None:
        raise ValueError("Please provide checkpoint path via +evaluation.checkpoint_path=...")

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
    gt_K = int(actions.size(1)) if actions.ndim == 3 else 1

    _log_info(f"[INFO] Robot state dim: {robot_state_dim}")
    _log_info(f"[INFO] Action dim: {action_dim}")
    _log_info(f"[INFO] GT chunk K (from dataset actions): {gt_K}")

    if debug_shapes:
        _log_info(f"[DEBUG] images shape={tuple(images.shape)}")
        _log_info(f"[DEBUG] point_clouds shape={tuple(point_clouds.shape)}")
        _log_info(f"[DEBUG] robot_states shape={tuple(robot_states.shape)}")
        if torch.is_tensor(raw_states):
            _log_info(f"[DEBUG] raw_states tensor shape={tuple(raw_states.shape)}")
        else:
            _log_info(f"[DEBUG] raw_states type={type(raw_states)}")
        _log_info(f"[DEBUG] actions shape={tuple(actions.shape)}")
        if is_pad is not None:
            _log_info(f"[DEBUG] is_pad shape={tuple(is_pad.shape)}")

    # model
    model = instantiate(
        config=config.agent.instantiate_config,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
    ).to(config.device)

    ckpt_summary = _load_checkpoint(model, ckpt_file=ckpt_file, device=config.device)
    _log_info(f"[INFO] ckpt load summary: missing={ckpt_summary['missing_keys_count']}, unexpected={ckpt_summary['unexpected_keys_count']}")

    model.eval()

    images = images.to(config.device)
    point_clouds = point_clouds.to(config.device)
    robot_states = robot_states.to(config.device)
    actions = actions.to(config.device)
    if is_pad is not None:
        is_pad = is_pad.to(config.device)

    # forward
    with torch.no_grad():
        try:
            preds = model(images, point_clouds, robot_states, texts, actions=actions, is_pad=is_pad)
        except TypeError:
            preds = model(images, point_clouds, robot_states, texts)

    a_pred = _get_actions_pred(preds)
    _log_info(f"[INFO] RAW model pred tensor shape: {tuple(a_pred.shape)}")

    # move to numpy
    a_pred_np = a_pred.detach().cpu().numpy()
    a_gt_np = actions.detach().cpu().numpy()
    pad_np = is_pad.detach().cpu().numpy()[0] if (is_pad is not None and is_pad.ndim >= 2) else None

    # --------- normalize to [K,A] if possible (but never tile a single-step pred) ----------
    mode = None

    if a_pred_np.ndim == 3:
        # [B,K,A]
        mode = "pred_is_chunk"
        a_pred_np = a_pred_np[0]
        a_gt_np = a_gt_np[0]
        if pad_np is not None:
            valid = ~pad_np.astype(bool)
            a_pred_np = a_pred_np[valid]
            a_gt_np = a_gt_np[valid]

    elif a_pred_np.ndim == 2:
        # could be [B,A] or already [K,A]
        if a_pred_np.shape[0] == gt_K and a_pred_np.shape[1] == action_dim:
            mode = "pred_is_chunk_2d"
            # here a_gt_np is [1,K,A], take [0]
            a_gt_np = a_gt_np[0]
        else:
            mode = "pred_is_single_step"
            _log_warn(
                f"Pred is 2D {a_pred_np.shape} but GT is chunk K={gt_K}. "
                "This indicates the model is single-step, not predicting 50 steps."
            )
            # save quick single-step proof and exit
            pred0 = a_pred_np[0]
            gt0 = a_gt_np[0, 0]
            mse0 = float(np.mean((pred0 - gt0) ** 2))
            mae0 = float(np.mean(np.abs(pred0 - gt0)))
            _log_info(f"[DIAG] single-step vs gt[0]: MSE={mse0:.6f}, MAE={mae0:.6f}")

            out = {
                "mode": mode,
                "gt_K": gt_K,
                "raw_pred_shape": list(a_pred.shape),
                "single_step_mse_vs_gt0": mse0,
                "single_step_mae_vs_gt0": mae0,
                "ckpt_summary": ckpt_summary,
            }
            with open(out_dir / f"plot_idx{plot_index:05d}_chunk_diagnosis.json", "w") as f:
                json.dump(out, f, indent=2)

            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(111)
            ax.plot(pred0, label="pred_step0", marker="o", alpha=0.9)
            ax.plot(gt0, label="gt_step0", marker="o", alpha=0.9)
            ax.set_title(f"Single-step pred vs gt[0] | idx={plot_index} | MSE={mse0:.6f} MAE={mae0:.6f}")
            ax.set_xlabel("action dim")
            ax.grid(True, alpha=0.3)
            ax.legend()
            p = out_dir / f"plot_idx{plot_index:05d}_single_step_vs_gt0.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            _log_info(f"[OK] Saved single-step plot: {colored(str(p), 'green')}")
            return

    else:
        raise ValueError(f"Unsupported pred shape: {a_pred_np.shape}")

    _log_info(f"[INFO] Mode: {mode} -> pred {a_pred_np.shape}, gt {a_gt_np.shape}")

    # --------- absolute + relative diagnosis ----------
    abs_diag = diagnose_chunk_absolute(a_pred_np, eps_abs=eps_abs)
    rel_diag = diagnose_chunk_relative_to_gt(a_pred_np, a_gt_np, repeat_ratio_thresh=repeat_ratio_thresh)

    _log_info(f"[DIAG-ABS] {json.dumps(abs_diag, indent=2)}")
    _log_info(f"[DIAG-REL] {json.dumps(rel_diag, indent=2)}")

    # hard proof prints
    _log_info(f"[PROOF] pred[0][:5]={a_pred_np[0, :5]}")
    if a_pred_np.shape[0] > 1:
        _log_info(f"[PROOF] pred[1][:5]={a_pred_np[1, :5]}")
    if a_pred_np.shape[0] > 2:
        _log_info(f"[PROOF] pred[2][:5]={a_pred_np[2, :5]}")
    _log_info(f"[PROOF] variation_ratio(pred/gt)={rel_diag['variation_ratio_pred_over_gt']:.6e}")
    _log_info(f"[PROOF] std_ratio_max={rel_diag['std_ratio_max']:.6e}, std_ratio_mean={rel_diag['std_ratio_mean']:.6e}")

    if abs_diag["looks_repeat_abs"] or (rel_diag["variation_ratio_pred_over_gt"] < repeat_ratio_thresh):
        _log_warn(
            "Pred changes over time are tiny compared to GT -> effectively repeating the same action across the 50-step chunk."
        )
    else:
        _log_info("Pred shows meaningful time variation relative to GT (not a flat chunk).")

    # --------- save plots ----------
    # 1) per-dim action traces
    action_plot_path = out_dir / f"plot_idx{plot_index:05d}_actions.png"
    plot_actions_1d(
        a_pred_np, a_gt_np,
        save_path=str(action_plot_path),
        title=f"Actions Pred vs GT | split={split} | idx={plot_index} | mode={mode}"
    )
    _log_info(f"[OK] Saved action plot: {colored(str(action_plot_path), 'green')}")

    # 2) pred-vs-gt chunk variation (the “is it flat?” smoking gun)
    var_path = out_dir / f"plot_idx{plot_index:05d}_chunk_variation_pred_vs_gt.png"
    plot_pred_gt_variation(
        a_pred_np, a_gt_np,
        save_path=str(var_path),
        title=(
            f"Chunk variation Pred vs GT | idx={plot_index} | "
            f"var_ratio={rel_diag['variation_ratio_pred_over_gt']:.2e} | "
            f"repeat_rel={rel_diag['looks_like_repeat_relative_to_gt']}"
        )
    )
    _log_info(f"[OK] Saved variation plot: {colored(str(var_path), 'green')}")

    # --------- save json diagnosis ----------
    out = {
        "mode": mode,
        "gt_K": int(gt_K),
        "raw_pred_shape": list(a_pred.shape),
        "abs_diag": abs_diag,
        "rel_diag": rel_diag,
        "ckpt_summary": ckpt_summary,
        "params": {
            "chunk_eps_abs": float(eps_abs),
            "repeat_ratio_thresh": float(repeat_ratio_thresh),
        },
    }
    with open(out_dir / f"plot_idx{plot_index:05d}_chunk_diagnosis.json", "w") as f:
        json.dump(out, f, indent=2)
    _log_info(f"[OK] Saved diagnosis json: {colored(str(out_dir / f'plot_idx{plot_index:05d}_chunk_diagnosis.json'), 'green')}")


if __name__ == "__main__":
    main()