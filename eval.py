#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import hydra
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hydra.utils import instantiate
from termcolor import colored

from scipy.ndimage import gaussian_filter1d

from omegaconf import OmegaConf

from lift3d.helpers.common import Logger, set_seed


# ============================================================
# Logger compat
# ============================================================
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
        print("-" * 110)


# ============================================================
# Batch helpers
# ============================================================
def _unpack_item(
    item,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor, Any, Optional[torch.Tensor]]:
    """
    Expected dataset item formats:
      - 6-tuple: images, point_clouds, robot_states, raw_states, actions, texts
      - 7-tuple: images, point_clouds, robot_states, raw_states, actions, texts, is_pad
    """
    if not isinstance(item, (tuple, list)):
        raise ValueError(f"Dataset item must be tuple/list, got {type(item)}")

    if len(item) == 6:
        images, point_clouds, robot_states, raw_states, actions, texts = item
        is_pad = None
    elif len(item) == 7:
        images, point_clouds, robot_states, raw_states, actions, texts, is_pad = item
    else:
        raise ValueError(f"Unexpected item size {len(item)}; expected 6 or 7.")

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

    raise ValueError(f"Unsupported model output type={type(preds)}")


def _ensure_chunk(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize to [B,K,A].
    Accepts:
      [B,A] -> [B,1,A]
      [B,K,A] -> unchanged
      [K,A] -> [1,K,A]
      [A] -> [1,1,A]
    """
    if x.dim() == 1:
        return x[None, None, :]
    if x.dim() == 2:
        return x[None, :, :]
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected action tensor, got {tuple(x.shape)}")


def _as_text_list(texts: Any, B: int) -> List[str]:
    if isinstance(texts, str):
        return [texts] * B
    if isinstance(texts, (list, tuple)):
        t = [str(x) for x in texts]
        if len(t) == B:
            return t
        if len(t) == 1:
            return t * B
        return [t[0]] * B
    return [str(texts)] * B


def _valid_prefix_len_from_is_pad(is_pad: Optional[Any], k: int) -> int:
    """
    Return a robust best-effort "valid prefix length" for a chunk, based on is_pad.
    Handles ambiguity of pad semantics by trying both interpretations and taking the longer valid prefix.
    """
    if is_pad is None:
        return k

    try:
        if torch.is_tensor(is_pad):
            pad_np = is_pad.detach().cpu().numpy()
        else:
            pad_np = np.array(is_pad)
        pad_np = pad_np.reshape(-1)[:k]
        pad_bool = pad_np.astype(bool)

        # interpretation 1: True means PAD
        if pad_bool.any():
            v1 = int(np.argmax(pad_bool))  # first True
        else:
            v1 = k

        # interpretation 2: True means VALID (so pad is ~pad_bool)
        pad2 = ~pad_bool
        if pad2.any():
            v2 = int(np.argmax(pad2))
        else:
            v2 = k

        v = max(v1, v2)
        v = int(np.clip(v, 0, k))
        if v == 0 and k > 0:
            return k
        return v
    except Exception:
        return k


def _scale_first_dims(x: np.ndarray, scale: float, pos_dim: int) -> np.ndarray:
    """Only scale first pos_dim dims (xyz)."""
    if scale == 1.0 or pos_dim <= 0:
        return x
    y = x.copy()
    y[..., :pos_dim] *= scale
    return y


def _scale_first_dims_torch(x: torch.Tensor, scale: float, pos_dim: int) -> torch.Tensor:
    """Torch version: only scale first pos_dim dims (xyz)."""
    if scale == 1.0 or pos_dim <= 0:
        return x
    y = x.clone()
    y[..., :pos_dim] *= scale
    return y


# ============================================================
# Checkpoint helpers
# ============================================================
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
    # NOTE: keep default torch.load behavior for maximum compatibility with older ckpts
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


# ============================================================
# Episode indexing
# ============================================================
@dataclass
class EpisodeStep:
    episode_id: int
    step_id: int


def _safe_to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if callable(v):
        return None
    if torch.is_tensor(v):
        if v.numel() != 1:
            return None
        return int(v.item())
    if isinstance(v, (np.generic,)):
        return int(v.item())
    if isinstance(v, (int, float, np.integer, np.floating)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except Exception:
            return None
    return None


def _default_infer_episode_step(dataset, index: int) -> EpisodeStep:
    """
    Best-effort inference from dataset[index][3] (raw_states).
    Avoid key 't' (torch.Tensor has .t()).
    """
    item = dataset[index]
    raw_states = item[3]

    ep_keys = ("episode_index", "episode_id", "episode")
    st_keys = ("step_index", "frame_index", "step")

    def get_key(obj: Any, keys: Tuple[str, ...]) -> Optional[int]:
        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    out = _safe_to_int(obj[k])
                    if out is not None:
                        return out
        for k in keys:
            if hasattr(obj, k):
                vv = getattr(obj, k)
                out = _safe_to_int(vv)
                if out is not None:
                    return out
        return None

    ep = get_key(raw_states, ep_keys)
    st = get_key(raw_states, st_keys)
    if ep is None or st is None:
        raise RuntimeError(
            "Cannot infer (episode_id, step_id) from raw_states. "
            "Please add dataset.index_to_episode_step(i)->(episode_id,step_id), "
            "or provide evaluation.episode_length for fallback segmentation."
        )
    return EpisodeStep(ep, st)


def _index_to_episode_step(dataset, index: int, episode_length_fallback: Optional[int]) -> EpisodeStep:
    if hasattr(dataset, "index_to_episode_step") and callable(getattr(dataset, "index_to_episode_step")):
        ep, st = dataset.index_to_episode_step(index)
        return EpisodeStep(int(ep), int(st))
    if hasattr(dataset, "get_episode_step") and callable(getattr(dataset, "get_episode_step")):
        ep, st = dataset.get_episode_step(index)
        return EpisodeStep(int(ep), int(st))

    try:
        return _default_infer_episode_step(dataset, index)
    except Exception:
        if episode_length_fallback is not None and episode_length_fallback > 0:
            ep = index // int(episode_length_fallback)
            st = index % int(episode_length_fallback)
            return EpisodeStep(int(ep), int(st))
        return EpisodeStep(0, int(index))


def _build_episode_index_map(dataset, episode_length_fallback: Optional[int]) -> Dict[int, List[int]]:
    tmp: Dict[int, List[Tuple[int, int]]] = {}
    for idx in range(len(dataset)):
        epst = _index_to_episode_step(dataset, idx, episode_length_fallback)
        tmp.setdefault(epst.episode_id, []).append((epst.step_id, idx))

    out: Dict[int, List[int]] = {}
    for ep, pairs in tmp.items():
        pairs.sort(key=lambda x: x[0])
        out[ep] = [i for _, i in pairs]
    return out


# ============================================================
# Math: smoothing / rot6 / rollout
# ============================================================
def _maybe_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return x
    return gaussian_filter1d(x, sigma=float(sigma), axis=0)


def _rot6_to_mat(rot6: np.ndarray) -> np.ndarray:
    """
    rot6: (..., 6) -> (..., 3, 3) via Gram-Schmidt
    """
    a1 = rot6[..., 0:3]
    a2 = rot6[..., 3:6]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)

    b3 = np.cross(b1, b2, axis=-1)

    R = np.stack([b1, b2, b3], axis=-1)  # (...,3,3) columns
    return R


def _mat_to_rot6(R: np.ndarray) -> np.ndarray:
    """
    R: (...,3,3) -> (...,6) taking first two columns
    """
    c1 = R[..., :, 0]
    c2 = R[..., :, 1]
    return np.concatenate([c1, c2], axis=-1)


def rollout_state_from_actions(
    init_state: np.ndarray,
    actions: np.ndarray,
    pos_dim: int = 3,
    rot6_slice: Tuple[int, int] = (3, 9),
    gripper_idx: Optional[int] = 9,
) -> np.ndarray:
    """
    Rollout state trajectory by applying delta actions sequentially.

    Assumption:
      state = [xyz(0:3), rot6(3:9), gripper(9)]  (dim=10)
      action = [dxyz, drot6, dgrip]              (dim=10)
      xyz_next = xyz + dxyz
      R_next = R_current @ R_delta
      gripper_next = gripper + dgrip
    """
    init_state = np.asarray(init_state, dtype=np.float32).copy()
    actions = np.asarray(actions, dtype=np.float32)
    T = actions.shape[0] + 1
    D = init_state.shape[0]
    out = np.zeros((T, D), dtype=np.float32)
    out[0] = init_state

    r0, r1 = rot6_slice
    for t in range(T - 1):
        s = out[t].copy()
        a = actions[t]

        s[:pos_dim] = s[:pos_dim] + a[:pos_dim]

        if r1 > r0 and r1 <= D and r1 <= a.shape[0]:
            R_cur = _rot6_to_mat(s[r0:r1])
            R_dlt = _rot6_to_mat(a[r0:r1])
            R_nxt = R_cur @ R_dlt
            s[r0:r1] = _mat_to_rot6(R_nxt)

        if gripper_idx is not None and gripper_idx < D and gripper_idx < a.shape[0]:
            s[gripper_idx] = s[gripper_idx] + a[gripper_idx]

        out[t + 1] = s

    return out


# ============================================================
# Plotting utilities (many figures)
# ============================================================
def _dim_labels_default(dim: int) -> List[str]:
    if dim == 10:
        return [
            "Position X", "Position Y", "Position Z",
            "Rotation 1", "Rotation 2", "Rotation 3",
            "Rotation 4", "Rotation 5", "Rotation 6",
            "Gripper",
        ]
    return [f"Dim {i}" for i in range(dim)]


def plot_series_pred_gt(
    pred: np.ndarray,
    gt: np.ndarray,
    save_path: str,
    title: str,
    labels: Optional[List[str]] = None,
):
    """
    pred/gt: [T, D]
    Robust: auto-align lengths (crop to min T).
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got pred{pred.shape}, gt{gt.shape}")
    D = min(pred.shape[1], gt.shape[1])
    T = min(pred.shape[0], gt.shape[0])
    pred = pred[:T, :D]
    gt = gt[:T, :D]

    if labels is None:
        labels = _dim_labels_default(D)
    else:
        labels = labels[:D]

    t = np.arange(T)

    fig, axes = plt.subplots(D, 1, figsize=(14, 2.2 * D), sharex=True)
    if D == 1:
        axes = [axes]
    fig.suptitle(title)

    for i in range(D):
        ax = axes[i]
        ax.plot(t, pred[:, i], label="Predicted")
        ax.plot(t, gt[:, i], label="Ground Truth")
        mse = float(np.mean((pred[:, i] - gt[:, i]) ** 2))
        mae = float(np.mean(np.abs(pred[:, i] - gt[:, i])))
        ax.set_title(f"{labels[i]} | MSE: {mse:.6f}, MAE: {mae:.6f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_series_error(
    pred: np.ndarray,
    gt: np.ndarray,
    save_path: str,
    title: str,
    labels: Optional[List[str]] = None,
    abs_err: bool = False,
):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got pred{pred.shape}, gt{gt.shape}")

    D = min(pred.shape[1], gt.shape[1])
    T = min(pred.shape[0], gt.shape[0])
    pred = pred[:T, :D]
    gt = gt[:T, :D]

    t = np.arange(T)
    if labels is None:
        labels = _dim_labels_default(D)
    else:
        labels = labels[:D]

    err = pred - gt
    if abs_err:
        err = np.abs(err)

    fig, axes = plt.subplots(D, 1, figsize=(14, 2.2 * D), sharex=True)
    if D == 1:
        axes = [axes]
    fig.suptitle(title)

    for i in range(D):
        ax = axes[i]
        ax.plot(t, err[:, i], label="|Pred-GT|" if abs_err else "Pred-GT")
        ax.set_title(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_error_histograms(
    pred: np.ndarray,
    gt: np.ndarray,
    save_path: str,
    title: str,
    pos_dim: int = 3,
    rot6_slice: Tuple[int, int] = (3, 9),
    gripper_idx: Optional[int] = 9,
):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got pred{pred.shape}, gt{gt.shape}")

    D = min(pred.shape[1], gt.shape[1])
    T = min(pred.shape[0], gt.shape[0])
    pred = pred[:T, :D]
    gt = gt[:T, :D]

    err = pred - gt

    groups = []
    group_names = []

    if pos_dim > 0 and D >= pos_dim:
        groups.append(err[:, :pos_dim].reshape(-1))
        group_names.append("xyz")

    r0, r1 = rot6_slice
    if r1 > r0 and D >= r1:
        groups.append(err[:, r0:r1].reshape(-1))
        group_names.append("rot6")

    if gripper_idx is not None and D > gripper_idx:
        groups.append(err[:, gripper_idx].reshape(-1))
        group_names.append("gripper")

    n = len(groups)
    if n == 0:
        _log_warn("[WARN] No valid groups for histogram; skip.")
        return

    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(title)

    for i, (g, name) in enumerate(zip(groups, group_names)):
        ax = axes[i]
        ax.hist(g, bins=80)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name} error hist | mean={float(np.mean(g)):.6f} std={float(np.std(g)):.6f}")
        ax.set_xlabel("error")
        ax.set_ylabel("count")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_pose_3d_and_gripper(
    pred_state: np.ndarray,
    gt_state: np.ndarray,
    save_prefix: str,
    sample_freq: int = 10,
    pos_dim: int = 3,
    rot6_slice: Tuple[int, int] = (3, 9),
    gripper_idx: Optional[int] = 9,
):
    pred_state = np.asarray(pred_state)
    gt_state = np.asarray(gt_state)
    if pred_state.ndim != 2 or gt_state.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got pred{pred_state.shape}, gt{gt_state.shape}")

    T = min(len(pred_state), len(gt_state))
    D = min(pred_state.shape[1], gt_state.shape[1])
    pred_state = pred_state[:T, :D]
    gt_state = gt_state[:T, :D]

    if D < pos_dim:
        _log_warn("[WARN] Not enough dims for 3D pose plot; skip.")
        return

    pred_pos = pred_state[:, :pos_dim]
    gt_pos = gt_state[:, :pos_dim]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors_pred = plt.cm.Blues(np.linspace(0.3, 1, T))
    colors_true = plt.cm.Reds(np.linspace(0.3, 1, T))

    for i in range(T - 1):
        ax.plot(pred_pos[i:i+2, 0], pred_pos[i:i+2, 1], pred_pos[i:i+2, 2], color=colors_pred[i], alpha=0.8)
        ax.plot(gt_pos[i:i+2, 0], gt_pos[i:i+2, 1], gt_pos[i:i+2, 2], color=colors_true[i], alpha=0.8)

    ax.scatter(pred_pos[0, 0], pred_pos[0, 1], pred_pos[0, 2], c="blue", marker="o", s=80, label="Pred Start")
    ax.scatter(pred_pos[-1, 0], pred_pos[-1, 1], pred_pos[-1, 2], c="blue", marker="s", s=80, label="Pred End")
    ax.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], c="red", marker="o", s=80, label="GT Start")
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], c="red", marker="s", s=80, label="GT End")

    r0, r1 = rot6_slice
    if r1 > r0 and D >= r1:
        sample_idx = np.arange(0, T, max(1, int(sample_freq)))
        scale = 0.02
        for i in sample_idx:
            p = pred_pos[i]
            g = gt_pos[i]

            Rp = _rot6_to_mat(pred_state[i, r0:r1])
            vp1, vp2 = Rp[:, 0], Rp[:, 1]
            ax.quiver(p[0], p[1], p[2], vp1[0], vp1[1], vp1[2], length=scale, color=colors_pred[i], alpha=0.35)
            ax.quiver(p[0], p[1], p[2], vp2[0], vp2[1], vp2[2], length=scale, color=colors_pred[i], alpha=0.35)

            Rg = _rot6_to_mat(gt_state[i, r0:r1])
            vg1, vg2 = Rg[:, 0], Rg[:, 1]
            ax.quiver(g[0], g[1], g[2], vg1[0], vg1[1], vg1[2], length=scale, color=colors_true[i], alpha=0.35)
            ax.quiver(g[0], g[1], g[2], vg2[0], vg2[1], vg2[2], length=scale, color=colors_true[i], alpha=0.35)

    pos_mse = float(np.mean((pred_pos - gt_pos) ** 2))
    rot_mse = float(np.mean((pred_state[:, r0:r1] - gt_state[:, r0:r1]) ** 2)) if (r1 > r0 and D >= r1) else 0.0
    ax.set_title(f"3D Pose Trajectory | Position MSE={pos_mse:.6f} | Rotation6D MSE={rot_mse:.6f}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.grid(True)
    ax.legend(loc="best")

    max_range = np.array([
        pred_pos.max(0) - pred_pos.min(0),
        gt_pos.max(0) - gt_pos.min(0)
    ]).max() / 2.0
    mid_x = (pred_pos[:, 0].mean() + gt_pos[:, 0].mean()) / 2
    mid_y = (pred_pos[:, 1].mean() + gt_pos[:, 1].mean()) / 2
    mid_z = (pred_pos[:, 2].mean() + gt_pos[:, 2].mean()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    fig.savefig(save_prefix + "_trajectory3d.png", dpi=220)
    plt.close(fig)

    if gripper_idx is not None and D > gripper_idx:
        fig2 = plt.figure(figsize=(12, 4))
        ax2 = fig2.add_subplot(111)
        ax2.plot(pred_state[:, gripper_idx], label="Predicted")
        ax2.plot(gt_state[:, gripper_idx], label="Ground Truth")
        ax2.set_title("Gripper Position")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Gripper")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")
        fig2.savefig(save_prefix + "_gripper.png", dpi=220)
        plt.close(fig2)


# ============================================================
# KEY: Disable conditional prior at inference (force z=0)
# ============================================================
def _infer_latent_dim_from_config_or_model(config, model: torch.nn.Module) -> Optional[int]:
    # 1) config paths (common names)
    for path in [
        "agent.instantiate_config.latent_dim",
        "agent.instantiate_config.z_dim",
        "agent.instantiate_config.latent_size",
        "agent.latent_dim",
        "agent.z_dim",
    ]:
        v = OmegaConf.select(config, path, default=None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass

    # 2) model attributes
    for attr in ["latent_dim", "z_dim", "latent_size", "nz", "dim_z"]:
        if hasattr(model, attr):
            try:
                return int(getattr(model, attr))
            except Exception:
                pass

    # 3) state_dict heuristic: look for a prior head with out_features=2*latent_dim
    try:
        sd = model.state_dict()
        cand = []
        for k, w in sd.items():
            if not torch.is_tensor(w) or w.ndim != 2:
                continue
            name = k.lower()
            if "prior" in name and "weight" in name:
                cand.append((k, w.shape))
        # pick the smallest "prior" linear
        if cand:
            cand.sort(key=lambda x: x[1][0])  # sort by out_features
            out_features = int(cand[0][1][0])
            # often prior outputs 2*latent_dim (mu+logvar)
            if out_features % 2 == 0:
                return out_features // 2
    except Exception:
        pass

    return None


def disable_conditional_prior_inference(model: torch.nn.Module, config) -> None:
    """
    Force inference (actions=None) to NOT depend on conditional prior.
    Implementation:
      - monkeypatch model._prior(global_d) to return mu=0, logvar=0
      - force model.sample_prior=False (so z = mu = 0, not random)
    This matches your old behavior (z=0) and avoids "conditional prior drift".
    """
    latent_dim = _infer_latent_dim_from_config_or_model(config, model)
    if latent_dim is None:
        _log_warn("[WARN] Could not infer latent_dim; cannot disable conditional prior safely. (Will keep original model behavior.)")
        return

    # turn off sampling from prior if the flag exists
    if hasattr(model, "sample_prior"):
        try:
            setattr(model, "sample_prior", False)
        except Exception:
            pass

    # patch _prior if present
    if hasattr(model, "_prior") and callable(getattr(model, "_prior")):
        def _prior_zero(global_d: torch.Tensor):
            mu = torch.zeros((global_d.shape[0], latent_dim), device=global_d.device, dtype=global_d.dtype)
            logvar = torch.zeros_like(mu)
            return mu, logvar
        try:
            model._prior = _prior_zero  # type: ignore[attr-defined]
            _log_info(f"[INFO] Disabled conditional prior: patched model._prior -> zeros (latent_dim={latent_dim}), sample_prior=False")
            return
        except Exception as e:
            _log_warn(f"[WARN] Failed to patch model._prior: {e}")

    # fallback: patch prior_net forward if exists
    if hasattr(model, "prior_net") and isinstance(getattr(model, "prior_net"), torch.nn.Module):
        prior_net = getattr(model, "prior_net")
        orig_forward = prior_net.forward

        def _forward_zero(x):
            B = x.shape[0]
            out = torch.zeros((B, 2 * latent_dim), device=x.device, dtype=x.dtype)
            return out

        try:
            prior_net.forward = _forward_zero  # type: ignore[method-assign]
            _log_info(f"[INFO] Disabled conditional prior: patched model.prior_net.forward -> zeros (latent_dim={latent_dim}), sample_prior=False")
            return
        except Exception as e:
            _log_warn(f"[WARN] Failed to patch model.prior_net.forward: {e}")

    _log_warn("[WARN] No _prior/prior_net found to patch. Conditional prior may still be used.")


# ============================================================
# Main
# ============================================================
@hydra.main(version_base=None, config_path="lift3d/config", config_name="train")
def main(config):
    _log_info(f"[INFO] Eval ACT (episode-stitch, FULL-CHUNK) + MANY PLOTS with {colored(pathlib.Path(__file__).absolute(), 'red')}")
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

    episode_id = getattr(eval_cfg, "episode_id", None) if eval_cfg is not None else None
    episode_length = getattr(eval_cfg, "episode_length", None) if eval_cfg is not None else None
    max_steps = getattr(eval_cfg, "max_steps", None) if eval_cfg is not None else None
    eval_batch_size = int(getattr(eval_cfg, "eval_batch_size", 16)) if eval_cfg is not None else 16
    chunk_stride_cfg = getattr(eval_cfg, "chunk_stride", None) if eval_cfg is not None else None

    train_pos_scale = float(getattr(eval_cfg, "pos_scale", 1.0)) if eval_cfg is not None else 1.0
    pos_dim = int(getattr(eval_cfg, "pos_dim", 3)) if eval_cfg is not None else 3
    unscale_to_meters = bool(getattr(eval_cfg, "unscale_to_meters", True)) if eval_cfg is not None else True

    smooth_sigma = float(getattr(eval_cfg, "smooth_sigma", 0.0)) if eval_cfg is not None else 0.0
    pose_sample_freq = int(getattr(eval_cfg, "pose_sample_freq", 10)) if eval_cfg is not None else 10

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    ckpt_file = _resolve_checkpoint_path(ckpt_path)
    _log_info(f"[INFO] Using checkpoint: {colored(ckpt_file, 'green')}")
    _log_info(f"[INFO] Split: {colored(split, 'green')}")
    _log_info(f"[INFO] eval_batch_size: {eval_batch_size}")

    dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        data_dir=config.dataset_dir,
        split=split,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split '{split}' is empty.")

    dataset_pos_scale = None
    if hasattr(dataset, "pos_scale"):
        try:
            dataset_pos_scale = float(getattr(dataset, "pos_scale"))
        except Exception:
            dataset_pos_scale = None

    if dataset_pos_scale is not None and dataset_pos_scale != 1.0:
        scale_for_model = 1.0
        model_space_scale = dataset_pos_scale
        _log_info(f"[INFO] Detected dataset.pos_scale={dataset_pos_scale} -> assume rs/actions already scaled for model.")
    else:
        scale_for_model = train_pos_scale
        model_space_scale = train_pos_scale
        _log_info(f"[INFO] Dataset has no pos_scale; using evaluation.pos_scale={train_pos_scale} for model inputs/GT.")

    inv_scale_for_plot = (1.0 / model_space_scale) if (unscale_to_meters and model_space_scale != 0.0) else 1.0
    _log_info(f"[INFO] pos_dim={pos_dim}, scale_for_model={scale_for_model}, model_space_scale={model_space_scale}, unscale_to_meters={unscale_to_meters}")
    _log_info(f"[INFO] smooth_sigma={smooth_sigma}, pose_sample_freq={pose_sample_freq}")

    it0 = dataset[0]
    images0, pc0, rs0, raw0, act0, txt0, pad0 = _unpack_item(it0)

    act0_t = act0 if torch.is_tensor(act0) else torch.as_tensor(act0, dtype=torch.float32)
    if act0_t.dim() == 2:
        action_dim = int(act0_t.shape[-1])
        gt_chunk_len0 = int(act0_t.shape[0])
    else:
        action_dim = int(act0_t.view(-1).shape[0])
        gt_chunk_len0 = 1

    robot_states0 = rs0 if torch.is_tensor(rs0) else torch.as_tensor(rs0, dtype=torch.float32)
    robot_state_dim = int(robot_states0.view(-1).shape[0])

    chunk_stride = int(chunk_stride_cfg) if chunk_stride_cfg is not None else int(gt_chunk_len0)
    chunk_stride = max(1, chunk_stride)

    _log_info(f"[INFO] Robot state dim: {robot_state_dim}")
    _log_info(f"[INFO] Action dim: {action_dim}")
    _log_info(f"[INFO] GT chunk len (from dataset[0]): {gt_chunk_len0}")
    _log_info(f"[INFO] chunk_stride: {chunk_stride}  (advance steps between chunks)")

    model = instantiate(
        config=config.agent.instantiate_config,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
    ).to(config.device)

    ckpt_summary = _load_checkpoint(model, ckpt_file)
    _log_info(f"[INFO] ckpt load summary: missing={ckpt_summary['missing_keys_count']}, unexpected={ckpt_summary['unexpected_keys_count']}")
    model.eval()

    # >>> IMPORTANT: disable conditional prior for inference (force z=0) <<<
    disable_conditional_prior_inference(model, config)

    ep_map = _build_episode_index_map(dataset, episode_length_fallback=episode_length)
    ep_ids = sorted(list(ep_map.keys()))
    if len(ep_ids) == 0:
        raise RuntimeError("No episodes found in dataset (episode map empty).")

    if episode_id is None:
        chosen_ep = ep_ids[0]
    else:
        chosen_ep = int(episode_id)
        if chosen_ep not in ep_map:
            raise ValueError(f"episode_id={chosen_ep} not found. available={ep_ids[:20]}... (total {len(ep_ids)})")

    idxs_all = ep_map[chosen_ep]
    if max_steps is not None:
        idxs_all = idxs_all[: int(max_steps)]

    if len(idxs_all) == 0:
        raise RuntimeError(f"Chosen episode {chosen_ep} has 0 steps after truncation.")

    epst_first = _index_to_episode_step(dataset, idxs_all[0], episode_length_fallback=episode_length)
    epst_last = _index_to_episode_step(dataset, idxs_all[-1], episode_length_fallback=episode_length)
    step_min_ep = int(epst_first.step_id)
    step_max_ep = int(epst_last.step_id)

    idxs = idxs_all[::chunk_stride]

    _log_info(f"[INFO] Chosen episode_id={chosen_ep}")
    _log_info(f"[INFO] episode steps (dataset items): {len(idxs_all)} | using chunk starts: {len(idxs)}")
    _log_info(f"[INFO] step_id range: {step_min_ep} .. {step_max_ep}")

    # ----------------------------
    # Build GT state trajectory from dataset steps
    # ----------------------------
    gt_state_raw_list: List[np.ndarray] = []
    gt_state_plot_list: List[np.ndarray] = []
    gt_state_stepid_list: List[int] = []

    for ds_idx in idxs_all:
        epst = _index_to_episode_step(dataset, ds_idx, episode_length_fallback=episode_length)
        item = dataset[ds_idx]
        _, _, robot_states, _, _, _, _ = _unpack_item(item)

        rs_t = robot_states if torch.is_tensor(robot_states) else torch.as_tensor(robot_states, dtype=torch.float32)
        if rs_t.dim() == 2 and rs_t.shape[0] == 1:
            rs_t = rs_t[0]
        rs_np = rs_t.detach().cpu().numpy().astype(np.float32)

        if rs_np.shape[0] < action_dim:
            break

        s_raw = rs_np[:action_dim].copy()
        if scale_for_model != 1.0:
            s_raw = _scale_first_dims(s_raw, scale_for_model, pos_dim)
        if inv_scale_for_plot != 1.0:
            s_plot = _scale_first_dims(s_raw, inv_scale_for_plot, pos_dim)
        else:
            s_plot = s_raw.copy()

        gt_state_raw_list.append(s_raw)
        gt_state_plot_list.append(s_plot)
        gt_state_stepid_list.append(int(epst.step_id))

    gt_state_available = (len(gt_state_raw_list) > 0 and gt_state_raw_list[0].shape[0] == action_dim)

    if gt_state_available:
        order_s = np.argsort(np.array(gt_state_stepid_list, dtype=np.int64))
        gt_state_raw = np.stack([gt_state_raw_list[i] for i in order_s], axis=0)
        gt_state_plot = np.stack([gt_state_plot_list[i] for i in order_s], axis=0)
        gt_state_steps = np.array([gt_state_stepid_list[i] for i in order_s], dtype=np.int64)
    else:
        gt_state_raw = None
        gt_state_plot = None
        gt_state_steps = None
        _log_warn("[WARN] Could not build GT state trajectory from robot_states (robot_state_dim < action_dim). Will skip state/3D plots.")

    # ----------------------------
    # Stitch FULL chunks into continuous action sequence
    # ----------------------------
    pred_list_raw: List[np.ndarray] = []
    gt_list_raw: List[np.ndarray] = []
    pred_list_plot: List[np.ndarray] = []
    gt_list_plot: List[np.ndarray] = []
    step_id_list: List[int] = []

    buf_imgs: List[torch.Tensor] = []
    buf_pcs: List[torch.Tensor] = []
    buf_rs: List[torch.Tensor] = []
    buf_txt: List[Any] = []
    buf_gt_chunk: List[torch.Tensor] = []
    buf_pad: List[Optional[torch.Tensor]] = []
    buf_stepid: List[int] = []

    def flush_batch():
        nonlocal buf_imgs, buf_pcs, buf_rs, buf_txt, buf_gt_chunk, buf_pad, buf_stepid
        if len(buf_imgs) == 0:
            return

        B = len(buf_imgs)
        device = config.device

        imgs = torch.stack(buf_imgs, dim=0).to(device)
        pcs = torch.stack(buf_pcs, dim=0).to(device)
        rs = torch.stack(buf_rs, dim=0).to(device)
        texts = _as_text_list(buf_txt, B)

        if scale_for_model != 1.0:
            rs = _scale_first_dims_torch(rs, scale_for_model, pos_dim)

        with torch.no_grad():
            # no actions passed -> inference path; we already patched conditional prior -> z=0
            preds_inf = model(imgs, pcs, rs, texts)

        a_inf = _ensure_chunk(_get_actions_pred(preds_inf))  # [B,K,A]
        a_inf_np = a_inf.detach().cpu().numpy()              # model space

        for bi in range(B):
            start_step = int(buf_stepid[bi])
            pred_chunk = a_inf_np[bi]                        # [K_pred,A] model space

            gt_chunk_t = buf_gt_chunk[bi]
            gt_chunk = _ensure_chunk(gt_chunk_t)[0].detach().cpu().numpy()  # [K_gt,A] dataset space

            if scale_for_model != 1.0:
                gt_chunk = _scale_first_dims(gt_chunk, scale_for_model, pos_dim)

            k = int(min(pred_chunk.shape[0], gt_chunk.shape[0]))
            if k <= 0:
                continue

            k_valid = _valid_prefix_len_from_is_pad(buf_pad[bi], k)

            remaining = (step_max_ep - start_step + 1)
            if remaining <= 0:
                continue
            k_valid = int(min(k_valid, remaining))
            if k_valid <= 0:
                continue

            for j in range(k_valid):
                p_raw = pred_chunk[j].copy()
                g_raw = gt_chunk[j].copy()

                pred_list_raw.append(p_raw)
                gt_list_raw.append(g_raw)

                if inv_scale_for_plot != 1.0:
                    p_plot = _scale_first_dims(p_raw, inv_scale_for_plot, pos_dim)
                    g_plot = _scale_first_dims(g_raw, inv_scale_for_plot, pos_dim)
                else:
                    p_plot = p_raw
                    g_plot = g_raw

                pred_list_plot.append(p_plot)
                gt_list_plot.append(g_plot)
                step_id_list.append(start_step + j)

        buf_imgs, buf_pcs, buf_rs, buf_txt, buf_gt_chunk, buf_pad, buf_stepid = [], [], [], [], [], [], []

    for ds_idx in idxs:
        epst = _index_to_episode_step(dataset, ds_idx, episode_length_fallback=episode_length)
        item = dataset[ds_idx]
        images, point_clouds, robot_states, _, actions, texts, is_pad = _unpack_item(item)

        img_t = images if torch.is_tensor(images) else torch.as_tensor(images, dtype=torch.float32)
        pc_t = point_clouds if torch.is_tensor(point_clouds) else torch.as_tensor(point_clouds, dtype=torch.float32)
        rs_t = robot_states if torch.is_tensor(robot_states) else torch.as_tensor(robot_states, dtype=torch.float32)
        act_t = actions if torch.is_tensor(actions) else torch.as_tensor(actions, dtype=torch.float32)

        if img_t.dim() == 4 and img_t.shape[0] == 1:
            img_t = img_t[0]
        if rs_t.dim() == 2 and rs_t.shape[0] == 1:
            rs_t = rs_t[0]
        if pc_t.dim() == 3 and pc_t.shape[0] == 1:
            pc_t = pc_t[0]

        buf_imgs.append(img_t)
        buf_pcs.append(pc_t)
        buf_rs.append(rs_t)
        buf_txt.append(texts)
        buf_gt_chunk.append(act_t)
        buf_pad.append(is_pad)
        buf_stepid.append(epst.step_id)

        if len(buf_imgs) >= eval_batch_size:
            flush_batch()

    flush_batch()

    if len(step_id_list) == 0:
        raise RuntimeError("No stitched actions produced. Check chunk_stride / is_pad / episode step ids.")

    order = np.argsort(np.array(step_id_list, dtype=np.int64))

    pred_raw = np.stack([pred_list_raw[i] for i in order], axis=0)
    gt_raw = np.stack([gt_list_raw[i] for i in order], axis=0)

    pred_plot = np.stack([pred_list_plot[i] for i in order], axis=0)
    gt_plot = np.stack([gt_list_plot[i] for i in order], axis=0)

    steps = np.array([step_id_list[i] for i in order], dtype=np.int64)

    pred_plot_s = _maybe_smooth(pred_plot, smooth_sigma)
    gt_plot_s = _maybe_smooth(gt_plot, smooth_sigma)

    T = pred_plot.shape[0]
    _log_info(f"[INFO] Built stitched FULL-CHUNK action trajectory: T={T}, A={pred_plot.shape[1]}")
    _log_info(f"[INFO] stitched step_id range: {int(steps.min())} .. {int(steps.max())}")

    mse_raw = float(np.mean((pred_raw - gt_raw) ** 2))
    mae_raw = float(np.mean(np.abs(pred_raw - gt_raw)))

    mse_plot = float(np.mean((pred_plot - gt_plot) ** 2))
    mae_plot = float(np.mean(np.abs(pred_plot - gt_plot)))

    _log_info(f"[EPISODE][RAW/model-space] MSE={mse_raw:.6f}  MAE={mae_raw:.6f}")
    if unscale_to_meters:
        _log_info(f"[EPISODE][PLOT/meters for xyz] MSE={mse_plot:.6f}  MAE={mae_plot:.6f}")
    else:
        _log_info(f"[EPISODE][PLOT same as raw] MSE={mse_plot:.6f}  MAE={mae_plot:.6f}")

    # ============================================================
    # MANY PLOTS (actions)
    # ============================================================
    labels = _dim_labels_default(pred_plot.shape[1])
    unit_note = "xyz in meters" if unscale_to_meters else "xyz in model-space"

    fig_actions = out_dir / f"episode_{chosen_ep}_ACTIONS_pred_vs_gt.png"
    plot_series_pred_gt(
        pred_plot_s, gt_plot_s,
        save_path=str(fig_actions),
        title=f"Actions Pred vs GT | {unit_note} | task={config.task_name} | split={split} | episode={chosen_ep} | T={T} | stride={chunk_stride}",
        labels=labels,
    )
    _log_info(f"[OK] Saved: {colored(str(fig_actions), 'green')}")

    fig_err = out_dir / f"episode_{chosen_ep}_ACTIONS_error_pred_minus_gt.png"
    plot_series_error(
        pred_plot_s, gt_plot_s,
        save_path=str(fig_err),
        title=f"Actions Error (Pred-GT) | {unit_note} | episode={chosen_ep}",
        labels=labels,
        abs_err=False,
    )
    _log_info(f"[OK] Saved: {colored(str(fig_err), 'green')}")

    fig_abserr = out_dir / f"episode_{chosen_ep}_ACTIONS_abs_error.png"
    plot_series_error(
        pred_plot_s, gt_plot_s,
        save_path=str(fig_abserr),
        title=f"Actions |Abs Error| | {unit_note} | episode={chosen_ep}",
        labels=labels,
        abs_err=True,
    )
    _log_info(f"[OK] Saved: {colored(str(fig_abserr), 'green')}")

    fig_hist = out_dir / f"episode_{chosen_ep}_ACTIONS_error_hist.png"
    plot_error_histograms(
        pred_plot, gt_plot,
        save_path=str(fig_hist),
        title=f"Actions Error Histograms | {unit_note} | episode={chosen_ep}",
        pos_dim=pos_dim,
        rot6_slice=(3, 9) if pred_plot.shape[1] >= 9 else (0, 0),
        gripper_idx=9 if pred_plot.shape[1] > 9 else None,
    )
    _log_info(f"[OK] Saved: {colored(str(fig_hist), 'green')}")

    # ============================================================
    # State trajectory plots (rollout) — FIXED LENGTH ALIGNMENT
    # ============================================================
    pred_state_raw = None
    pred_state_plot = None
    pred_state_steps = None

    if gt_state_available and gt_state_raw is not None and gt_state_plot is not None and gt_state_steps is not None:
        step_to_action_idx = {int(s): i for i, s in enumerate(steps.tolist())}

        common_steps = [int(s) for s in gt_state_steps.tolist() if int(s) in step_to_action_idx]
        common_steps.sort()

        if len(common_steps) >= 1:
            start_step = common_steps[0]
            gt_start_i = int(np.where(gt_state_steps == start_step)[0][0])
            act_start_i = step_to_action_idx[start_step]

            remaining_gt = int(gt_state_raw.shape[0] - gt_start_i)
            if remaining_gt <= 1:
                _log_warn("[WARN] Not enough GT state remaining to rollout; skip state plots.")
            else:
                max_actions = remaining_gt - 1
                actions_for_rollout_raw = pred_raw[act_start_i: act_start_i + max_actions]  # <= max_actions

                init_s_raw = gt_state_raw[gt_start_i].copy()

                pred_state_raw = rollout_state_from_actions(
                    init_state=init_s_raw,
                    actions=actions_for_rollout_raw,
                    pos_dim=pos_dim,
                    rot6_slice=(3, 9) if action_dim >= 9 else (0, 0),
                    gripper_idx=9 if action_dim > 9 else None,
                )

                L = int(pred_state_raw.shape[0])
                gt_state_raw_seg = gt_state_raw[gt_start_i: gt_start_i + L]
                gt_state_plot_seg = gt_state_plot[gt_start_i: gt_start_i + L]

                if inv_scale_for_plot != 1.0:
                    pred_state_plot = pred_state_raw.copy()
                    pred_state_plot[:, :pos_dim] *= inv_scale_for_plot
                else:
                    pred_state_plot = pred_state_raw.copy()

                pred_state_plot_s = _maybe_smooth(pred_state_plot, smooth_sigma)
                gt_state_plot_s = _maybe_smooth(gt_state_plot_seg, smooth_sigma)
                pred_state_steps = gt_state_steps[gt_start_i: gt_start_i + L]

                fig_state = out_dir / f"episode_{chosen_ep}_STATE_pred_vs_gt.png"
                plot_series_pred_gt(
                    pred_state_plot_s, gt_state_plot_s,
                    save_path=str(fig_state),
                    title=f"State Trajectory Pred vs GT (rollout from actions) | {unit_note} | episode={chosen_ep} | T={pred_state_plot_s.shape[0]}",
                    labels=_dim_labels_default(min(action_dim, pred_state_plot_s.shape[1])),
                )
                _log_info(f"[OK] Saved: {colored(str(fig_state), 'green')}")

                save_prefix = str(out_dir / f"episode_{chosen_ep}_POSE")
                plot_pose_3d_and_gripper(
                    pred_state_plot_s, gt_state_plot_s,
                    save_prefix=save_prefix,
                    sample_freq=pose_sample_freq,
                    pos_dim=pos_dim,
                    rot6_slice=(3, 9) if action_dim >= 9 else (0, 0),
                    gripper_idx=9 if action_dim > 9 else None,
                )
                _log_info(f"[OK] Saved: {colored(save_prefix + '_trajectory3d.png', 'green')}")
                if action_dim > 9:
                    _log_info(f"[OK] Saved: {colored(save_prefix + '_gripper.png', 'green')}")
        else:
            _log_warn("[WARN] No overlapping steps between GT state steps and stitched action steps; skip state plots.")

    # ============================================================
    # Save NPZ
    # ============================================================
    npz_path = out_dir / f"episode_{chosen_ep}_FULLPLOTS_eval_outputs.npz"
    np.savez_compressed(
        str(npz_path),
        pred_plot=pred_plot,
        gt_plot=gt_plot,
        pred_plot_s=pred_plot_s,
        gt_plot_s=gt_plot_s,
        pred_raw=pred_raw,
        gt_raw=gt_raw,
        step_id=steps,
        pred_state_raw=pred_state_raw if pred_state_raw is not None else np.zeros((0, action_dim), dtype=np.float32),
        pred_state_plot=pred_state_plot if pred_state_plot is not None else np.zeros((0, action_dim), dtype=np.float32),
        pred_state_step_id=pred_state_steps if pred_state_steps is not None else np.zeros((0,), dtype=np.int64),
        gt_state_raw=gt_state_raw if gt_state_raw is not None else np.zeros((0, action_dim), dtype=np.float32),
        gt_state_plot=gt_state_plot if gt_state_plot is not None else np.zeros((0, action_dim), dtype=np.float32),
        gt_state_step_id=gt_state_steps if gt_state_steps is not None else np.zeros((0,), dtype=np.int64),
        episode_id=int(chosen_ep),
        split=split,
        checkpoint=str(ckpt_file),
        chunk_stride=int(chunk_stride),
        pos_dim=int(pos_dim),
        model_space_scale=float(model_space_scale),
        unscale_to_meters=bool(unscale_to_meters),
        inv_scale_for_plot=float(inv_scale_for_plot),
        smooth_sigma=float(smooth_sigma),
    )
    _log_info(f"[OK] Saved npz: {colored(str(npz_path), 'green')}")

    summary = {
        "task": config.task_name,
        "split": split,
        "checkpoint": str(ckpt_file),
        "episode_id": int(chosen_ep),
        "T_actions": int(T),
        "A": int(pred_plot.shape[1]),
        "metrics": {
            "mse_raw": mse_raw,
            "mae_raw": mae_raw,
            "mse_plot": mse_plot,
            "mae_plot": mae_plot,
        },
        "stitched_step_id_min": int(steps.min()) if len(steps) else None,
        "stitched_step_id_max": int(steps.max()) if len(steps) else None,
        "episode_step_id_min": int(step_min_ep),
        "episode_step_id_max": int(step_max_ep),
        "ckpt_summary": {
            "missing_keys_count": ckpt_summary["missing_keys_count"],
            "unexpected_keys_count": ckpt_summary["unexpected_keys_count"],
        },
        "scaling": {
            "pos_dim": int(pos_dim),
            "dataset_pos_scale_detected": None if dataset_pos_scale is None else float(dataset_pos_scale),
            "evaluation_pos_scale": float(train_pos_scale),
            "scale_for_model_applied_in_eval": float(scale_for_model),
            "model_space_scale": float(model_space_scale),
            "unscale_to_meters": bool(unscale_to_meters),
            "inv_scale_for_plot": float(inv_scale_for_plot),
        },
        "plots": {
            "smooth_sigma": float(smooth_sigma),
            "pose_sample_freq": int(pose_sample_freq),
        },
        "params": {
            "eval_batch_size": int(eval_batch_size),
            "episode_length_fallback": None if episode_length is None else int(episode_length),
            "max_steps": None if max_steps is None else int(max_steps),
            "chunk_stride": int(chunk_stride),
        },
        "state_plots_enabled": bool(gt_state_available),
        "inference_latent": "forced_zero (conditional prior disabled)",
    }
    out_json = out_dir / "eval_episode_stitch_summary_FULLPLOTS.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    _log_info(f"[OK] Saved summary json: {colored(str(out_json), 'green')}")

    _print_sep()
    _log_info(f"[DONE] Output dir: {colored(str(out_dir), 'green')}")


if __name__ == "__main__":
    main()