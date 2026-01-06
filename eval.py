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
        print("-" * 110)


# ----------------------------
# Batch helpers
# ----------------------------
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
      [K,A] -> [1,K,A]
      [A] -> [1,1,A]
    """
    if x.dim() == 1:
        return x[None, None, :]
    if x.dim() == 2:
        # ambiguous: could be [B,A] or [K,A]; in our usage for GT actions, usually [K,A]
        return x[None, :, :]
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected action tensor, got {tuple(x.shape)}")


def _as_text_list(texts: Any, B: int) -> List[str]:
    if isinstance(texts, str):
        return [texts] * B
    if isinstance(texts, (list, tuple)):
        # DataLoader(batch_size>1) often gives list of strings already
        t = [str(x) for x in texts]
        if len(t) == B:
            return t
        if len(t) == 1:
            return t * B
        # fallback
        return [t[0]] * B
    return [str(texts)] * B


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
# Episode indexing (关键：把 dataset index -> (episode, step))
# ----------------------------
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
        # 最后兜底：全当一个 episode
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


# ----------------------------
# Plot: 700-step continuous actions
# ----------------------------
def plot_trajectory_1episode(pred: np.ndarray, gt: np.ndarray, save_path: str, title: str):
    """
    pred/gt: [T, A]
    """
    T, A = pred.shape
    t = np.arange(T)

    fig, axes = plt.subplots(A, 1, figsize=(14, 2.2 * A), sharex=True)
    if A == 1:
        axes = [axes]
    fig.suptitle(title)

    for i in range(A):
        ax = axes[i]
        ax.plot(t, pred[:, i], label="Predicted")
        ax.plot(t, gt[:, i], label="Ground Truth")
        mse = float(np.mean((pred[:, i] - gt[:, i]) ** 2))
        mae = float(np.mean(np.abs(pred[:, i] - gt[:, i])))
        ax.set_title(f"MSE: {mse:.6f}, MAE: {mae:.6f}")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(f"Action {i}")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
@hydra.main(version_base=None, config_path="lift3d/config", config_name="train")
def main(config):
    _log_info(f"[INFO] Eval ACT (episode-stitch) with {colored(pathlib.Path(__file__).absolute(), 'red')}")
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

    # episode stitch controls
    episode_id = getattr(eval_cfg, "episode_id", None) if eval_cfg is not None else None  # int or None
    episode_length = getattr(eval_cfg, "episode_length", None) if eval_cfg is not None else None  # fallback only
    max_steps = getattr(eval_cfg, "max_steps", None) if eval_cfg is not None else None  # optional truncate
    eval_batch_size = int(getattr(eval_cfg, "eval_batch_size", 16)) if eval_cfg is not None else 16

    # make inference deterministic (if your actor samples z)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    ckpt_file = _resolve_checkpoint_path(ckpt_path)
    _log_info(f"[INFO] Using checkpoint: {colored(ckpt_file, 'green')}")
    _log_info(f"[INFO] Split: {colored(split, 'green')}")
    _log_info(f"[INFO] eval_batch_size: {eval_batch_size}")

    # dataset (direct iteration, no DataLoader collation issues for raw_states)
    dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        data_dir=config.dataset_dir,
        split=split,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split '{split}' is empty.")

    # infer dims from first item
    it0 = dataset[0]
    images0, pc0, rs0, raw0, act0, txt0, pad0 = _unpack_item(it0)

    # action_dim from GT (chunk or single)
    act0_t = act0 if torch.is_tensor(act0) else torch.as_tensor(act0, dtype=torch.float32)
    if act0_t.dim() == 2:
        action_dim = int(act0_t.shape[-1])
    else:
        action_dim = int(act0_t.view(-1).shape[0])

    robot_states0 = rs0 if torch.is_tensor(rs0) else torch.as_tensor(rs0, dtype=torch.float32)
    robot_state_dim = int(robot_states0.view(-1).shape[0])

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

    # build episode map
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

    idxs = ep_map[chosen_ep]
    if max_steps is not None:
        idxs = idxs[: int(max_steps)]

    _log_info(f"[INFO] Chosen episode_id={chosen_ep}, num_steps={len(idxs)}")

    # We will build continuous arrays: pred_inf[t], gt[t]
    pred_list: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []
    step_id_list: List[int] = []

    # batched inference buffers
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

        imgs = torch.stack(buf_imgs, dim=0).to(device)                # [B,C,H,W] or [B,...]
        pcs = torch.stack(buf_pcs, dim=0).to(device)                  # [B,N,3] (assume fixed N)
        rs = torch.stack(buf_rs, dim=0).to(device)                    # [B,D]
        texts = _as_text_list(buf_txt, B)

        # inference
        with torch.no_grad():
            preds_inf = model(imgs, pcs, rs, texts)
        a_inf = _ensure_chunk(_get_actions_pred(preds_inf))           # [B,K,A]
        a0_inf = a_inf[:, 0, :].detach().cpu().numpy()                # [B,A]

        # GT: take first step of GT chunk
        for bi in range(B):
            gt_chunk = buf_gt_chunk[bi]                               # [K,A] or [A]
            gt_chunk = _ensure_chunk(gt_chunk)[0]                     # [K,A]
            gt0 = gt_chunk[0].detach().cpu().numpy()                  # [A]

            pred_list.append(a0_inf[bi])
            gt_list.append(gt0)
            step_id_list.append(int(buf_stepid[bi]))

        # reset
        buf_imgs, buf_pcs, buf_rs, buf_txt, buf_gt_chunk, buf_pad, buf_stepid = [], [], [], [], [], [], []

    # iterate steps in episode
    for ds_idx in idxs:
        epst = _index_to_episode_step(dataset, ds_idx, episode_length_fallback=episode_length)
        item = dataset[ds_idx]
        images, point_clouds, robot_states, raw_states, actions, texts, is_pad = _unpack_item(item)

        # tensors
        img_t = images if torch.is_tensor(images) else torch.as_tensor(images, dtype=torch.float32)
        pc_t = point_clouds if torch.is_tensor(point_clouds) else torch.as_tensor(point_clouds, dtype=torch.float32)
        rs_t = robot_states if torch.is_tensor(robot_states) else torch.as_tensor(robot_states, dtype=torch.float32)
        act_t = actions if torch.is_tensor(actions) else torch.as_tensor(actions, dtype=torch.float32)

        # ensure shapes: img [C,H,W], pc [N,3], rs [D]
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

    # sort by step_id (in case dataset order isn't perfect)
    order = np.argsort(np.array(step_id_list, dtype=np.int64))
    pred = np.stack([pred_list[i] for i in order], axis=0)  # [T,A]
    gt = np.stack([gt_list[i] for i in order], axis=0)      # [T,A]
    steps = np.array([step_id_list[i] for i in order], dtype=np.int64)

    # if steps are not 0..T-1, we still plot by index; also save step ids
    T = pred.shape[0]
    _log_info(f"[INFO] Built stitched trajectory: T={T}, A={pred.shape[1]}")
    _log_info(f"[INFO] step_id range: {steps.min()} .. {steps.max()}")

    # global metrics on the episode
    mse_all = float(np.mean((pred - gt) ** 2))
    mae_all = float(np.mean(np.abs(pred - gt)))
    _log_info(f"[EPISODE] MSE={mse_all:.6f}  MAE={mae_all:.6f}")

    # plot
    fig_path = out_dir / f"episode_{chosen_ep}_actions_pred_vs_gt.png"
    plot_trajectory_1episode(
        pred, gt,
        save_path=str(fig_path),
        title=f"Predicted vs Ground Truth Actions | task={config.task_name} | split={split} | episode={chosen_ep} | T={T}"
    )
    _log_info(f"[OK] Saved plot: {colored(str(fig_path), 'green')}")

    # save npz
    npz_path = out_dir / f"episode_{chosen_ep}_actions_pred_vs_gt.npz"
    np.savez_compressed(
        str(npz_path),
        pred=pred,
        gt=gt,
        step_id=steps,
        episode_id=int(chosen_ep),
        split=split,
        checkpoint=str(ckpt_file),
    )
    _log_info(f"[OK] Saved npz: {colored(str(npz_path), 'green')}")

    # summary json
    summary = {
        "task": config.task_name,
        "split": split,
        "checkpoint": str(ckpt_file),
        "episode_id": int(chosen_ep),
        "T": int(T),
        "A": int(pred.shape[1]),
        "mse": mse_all,
        "mae": mae_all,
        "step_id_min": int(steps.min()) if len(steps) else None,
        "step_id_max": int(steps.max()) if len(steps) else None,
        "ckpt_summary": {
            "missing_keys_count": ckpt_summary["missing_keys_count"],
            "unexpected_keys_count": ckpt_summary["unexpected_keys_count"],
        },
        "params": {
            "eval_batch_size": int(eval_batch_size),
            "episode_length_fallback": None if episode_length is None else int(episode_length),
            "max_steps": None if max_steps is None else int(max_steps),
        },
    }
    out_json = out_dir / "eval_episode_stitch_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    _log_info(f"[OK] Saved summary json: {colored(str(out_json), 'green')}")

    _print_sep()
    _log_info(f"[DONE] Output dir: {colored(str(out_dir), 'green')}")


if __name__ == "__main__":
    main()