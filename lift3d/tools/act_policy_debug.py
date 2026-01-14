#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import time
from typing import Optional, Tuple, Any, Dict, List

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from termcolor import colored

from lift3d.helpers.common import Logger, WandBLogger, set_seed
from lift3d.helpers.pytorch import AverageMeter, log_params_to_file


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


# ----------------------------
# batch + preds helpers
# ----------------------------
def _unpack_batch(batch) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor, Any, Optional[torch.Tensor]
]:
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


def _to_item(v):
    if torch.is_tensor(v):
        return v.item()
    return v


def _get_actions_pred(preds) -> torch.Tensor:
    """
    Extract action prediction tensor from model output.
    Supports:
      - Tensor
      - dict with 'actions'/'a_hat'
      - object with .actions/.a_hat
    """
    if torch.is_tensor(preds):
        return preds

    if isinstance(preds, dict):
        if "actions" in preds and torch.is_tensor(preds["actions"]):
            return preds["actions"]
        if "a_hat" in preds and torch.is_tensor(preds["a_hat"]):
            return preds["a_hat"]

    for k in ["actions", "a_hat"]:
        if hasattr(preds, k):
            v = getattr(preds, k)
            if torch.is_tensor(v):
                return v

    raise ValueError(f"Cannot find action prediction in preds type={type(preds)}")


def _ensure_chunk(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize to [B,K,A].
    Accepts:
      [B,A] -> [B,1,A]
      [B,K,A] -> unchanged
      [K,A] -> [1,K,A]
      [A] -> [1,1,A]
    """
    if x.ndim == 1:
        return x[None, None, :]
    if x.ndim == 2:
        # treat as [B,A] (safe for your pipeline)
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected tensor with ndim in {1,2,3}, got shape={tuple(x.shape)}")


def _flatten_valid_mask(is_pad: Optional[torch.Tensor], actions_chunk: torch.Tensor) -> Optional[torch.Tensor]:
    """
    is_pad expected [B,K] or [B,K,1]. Returns valid mask [B,K,1] float.
    Assumption: True means PAD.
    """
    if is_pad is None:
        return None
    if not torch.is_tensor(is_pad):
        is_pad = torch.as_tensor(is_pad)
    valid = (~is_pad.bool()).to(actions_chunk.dtype)
    if valid.ndim == 2:
        valid = valid.unsqueeze(-1)
    return valid


# ----------------------------
# metrics helpers
# ----------------------------
def _safe_stats(x: torch.Tensor) -> Dict[str, float]:
    x_det = x.detach()
    return {
        "mean": float(x_det.mean().item()),
        "std": float(x_det.std(unbiased=False).item()),
        "min": float(x_det.min().item()),
        "max": float(x_det.max().item()),
    }


def _masked_mse_mae(pred: torch.Tensor, gt: torch.Tensor, mask_valid: Optional[torch.Tensor]) -> Dict[str, float]:
    pred = _ensure_chunk(pred)
    gt = _ensure_chunk(gt)

    K = min(pred.shape[1], gt.shape[1])
    pred = pred[:, :K, :]
    gt = gt[:, :K, :]

    diff = pred - gt
    if mask_valid is not None:
        mask_valid = mask_valid[:, :K, :]
        diff = diff * mask_valid
        denom = mask_valid.sum().clamp_min(1.0) * pred.shape[-1]
        mse = (diff.pow(2).sum() / denom).item()
        mae = (diff.abs().sum() / denom).item()
    else:
        mse = diff.pow(2).mean().item()
        mae = diff.abs().mean().item()

    return {"mse": float(mse), "mae": float(mae), "K_used": int(K)}


def _cuda_mem() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_alloc_gb": float(torch.cuda.memory_allocated() / (1024**3)),
        "cuda_reserved_gb": float(torch.cuda.memory_reserved() / (1024**3)),
    }


def _write_jsonl(path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _maybe_mirror_write_jsonl(primary_path: str, mirror_dir: Optional[str], row: Dict[str, Any]):
    _write_jsonl(primary_path, row)
    if mirror_dir:
        mirror_path = os.path.join(mirror_dir, os.path.basename(primary_path))
        _write_jsonl(mirror_path, row)


def _maybe_forward_train(model, images, point_clouds, robot_states, texts, actions, is_pad):
    """Teacher-forcing forward for loss: try passing actions/is_pad if supported."""
    try:
        return model(images, point_clouds, robot_states, texts, actions=actions, is_pad=is_pad)
    except TypeError:
        try:
            return model(images, point_clouds, robot_states, texts, actions=actions)
        except TypeError:
            return model(images, point_clouds, robot_states, texts)


def _forward_open_loop(model, images, point_clouds, robot_states, texts):
    """Open-loop chunk prediction: do NOT pass actions/is_pad."""
    return model(images, point_clouds, robot_states, texts)


def _pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# ----------------------------
# checkpoint helpers (NEW)
# ----------------------------
def _atomic_torch_save(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, global_iter: int, extra: Optional[Dict[str, Any]] = None):
    ckpt = {
        "epoch": int(epoch),
        "global_iter": int(global_iter),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    if extra:
        ckpt["extra"] = extra
    _atomic_torch_save(ckpt, path)


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    valid_loader,
    loss_func,
    device: str,
    kl_weight: float,
    max_batches: Optional[int] = None,
    log_horizon_curve: bool = True,
) -> Dict[str, Any]:
    """
    Validation includes:
      1) teacher-forcing val loss (same as training loss)
      2) open-loop chunk MSE/MAE (no teacher forcing)
      3) pad_ratio vs batch_loss correlation
    """
    model.eval()

    loss_meter = AverageMeter()
    tf_mse_meter = AverageMeter()
    tf_mae_meter = AverageMeter()

    ol_mse_meter = AverageMeter()
    ol_mae_meter = AverageMeter()

    pad_ratios: List[float] = []
    batch_losses: List[float] = []

    ol_h_mse_sum: Optional[np.ndarray] = None
    ol_h_mae_sum: Optional[np.ndarray] = None
    ol_h_denom: Optional[np.ndarray] = None

    for bi, batch in enumerate(valid_loader):
        if max_batches is not None and bi >= int(max_batches):
            break

        images, point_clouds, robot_states, raw_states, actions, texts, is_pad = _unpack_batch(batch)

        images = images.to(device, non_blocking=True)
        point_clouds = point_clouds.to(device, non_blocking=True)
        robot_states = robot_states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        if is_pad is not None:
            is_pad = is_pad.to(device, non_blocking=True)

        # 1) teacher-forcing val loss
        preds_tf = _maybe_forward_train(model, images, point_clouds, robot_states, texts, actions, is_pad)

        if is_pad is not None:
            try:
                loss_result = loss_func(preds_tf, actions, is_pad=is_pad, kl_weight=kl_weight)
            except TypeError:
                try:
                    loss_result = loss_func(preds_tf, actions, is_pad=is_pad)
                except TypeError:
                    loss_result = loss_func(preds_tf, actions)
        else:
            try:
                loss_result = loss_func(preds_tf, actions, kl_weight=kl_weight)
            except TypeError:
                loss_result = loss_func(preds_tf, actions)

        loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
        loss_val = float(loss.item())
        loss_meter.update(loss_val, actions.shape[0])

        a_tf = _ensure_chunk(_get_actions_pred(preds_tf).float())
        a_gt = _ensure_chunk(actions.float())
        mask_valid = _flatten_valid_mask(is_pad, a_gt)

        tf_mm = _masked_mse_mae(a_tf, a_gt, mask_valid)
        tf_mse_meter.update(tf_mm["mse"], actions.shape[0])
        tf_mae_meter.update(tf_mm["mae"], actions.shape[0])

        # 2) open-loop chunk mse/mae
        preds_ol = _forward_open_loop(model, images, point_clouds, robot_states, texts)
        a_ol = _ensure_chunk(_get_actions_pred(preds_ol).float())

        K = min(a_ol.shape[1], a_gt.shape[1])
        a_ol = a_ol[:, :K, :]
        a_gt_k = a_gt[:, :K, :]

        mask_valid_k = _flatten_valid_mask(is_pad, a_gt_k)
        ol_mm = _masked_mse_mae(a_ol, a_gt_k, mask_valid_k)
        ol_mse_meter.update(ol_mm["mse"], actions.shape[0])
        ol_mae_meter.update(ol_mm["mae"], actions.shape[0])

        # horizon curves
        if log_horizon_curve and K > 1:
            diff = (a_ol - a_gt_k)
            if mask_valid_k is not None:
                mv = mask_valid_k  # [B,K,1]
                denom_h = mv.sum(dim=(0, 2)).detach().cpu().numpy().astype(np.float64)
                denom_h = np.maximum(denom_h, 1e-12)
                mse_h = ((diff * mv).pow(2).sum(dim=(0, 2)).detach().cpu().numpy().astype(np.float64) / denom_h)
                mae_h = ((diff * mv).abs().sum(dim=(0, 2)).detach().cpu().numpy().astype(np.float64) / denom_h)
                weight_h = denom_h
            else:
                mse_h = diff.pow(2).mean(dim=(0, 2)).detach().cpu().numpy().astype(np.float64)
                mae_h = diff.abs().mean(dim=(0, 2)).detach().cpu().numpy().astype(np.float64)
                weight_h = np.ones((K,), dtype=np.float64) * float(diff.shape[0])

            if ol_h_mse_sum is None:
                ol_h_mse_sum = np.zeros((K,), dtype=np.float64)
                ol_h_mae_sum = np.zeros((K,), dtype=np.float64)
                ol_h_denom = np.zeros((K,), dtype=np.float64)

            Kc = min(len(ol_h_mse_sum), K)
            ol_h_mse_sum[:Kc] += mse_h[:Kc] * weight_h[:Kc]
            ol_h_mae_sum[:Kc] += mae_h[:Kc] * weight_h[:Kc]
            ol_h_denom[:Kc] += weight_h[:Kc]

        # 3) pad vs loss
        if is_pad is not None:
            pad_ratio = float(is_pad.float().mean().item())
            pad_ratios.append(pad_ratio)
            batch_losses.append(loss_val)

    pad_corr = _pearson_corr(pad_ratios, batch_losses)

    out: Dict[str, Any] = {
        "val_loss": float(loss_meter.avg),
        "val_tf_mse": float(tf_mse_meter.avg),
        "val_tf_mae": float(tf_mae_meter.avg),
        "val_open_loop_mse": float(ol_mse_meter.avg),
        "val_open_loop_mae": float(ol_mae_meter.avg),
        "val_pad_mean": float(np.mean(pad_ratios)) if len(pad_ratios) else None,
        "val_pad_loss_corr": pad_corr,
        "val_num_batches": int(min(len(valid_loader), max_batches) if max_batches is not None else len(valid_loader)),
    }

    if ol_h_mse_sum is not None and ol_h_denom is not None:
        denom = np.maximum(ol_h_denom, 1e-12)
        out["val_open_loop_horizon_mse_curve"] = (ol_h_mse_sum / denom).astype(np.float64).tolist()
        out["val_open_loop_horizon_mae_curve"] = (ol_h_mae_sum / denom).astype(np.float64).tolist()

    return out


def _instantiate_base_dataset(base_cfg, data_root: str, split: str, task_name: Optional[str], camera_name: Optional[str], image_size: int):
    attempts = [
        dict(data_dir=data_root, split=split),
        dict(data_dir=data_root, split=split, camera_name=camera_name, image_size=image_size),
        dict(data_dir=data_root, split=split, task_name=task_name, camera_name=camera_name, image_size=image_size),
        dict(data_dir=data_root),
        dict(dataset_dir=data_root, split=split),
        dict(dataset_dir=data_root),
    ]

    last_err = None
    for kw in attempts:
        kw = {k: v for k, v in kw.items() if v is not None}
        try:
            return instantiate(base_cfg, **kw)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to instantiate base_dataset with any known signature. Last error: {last_err}")


# ----------------------------
# main
# ----------------------------
@hydra.main(version_base=None, config_path="../config", config_name="train_policy")
def main(config):
    set_seed(int(getattr(config, "seed", 0)))

    local_run_output_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    os.makedirs(local_run_output_dir, exist_ok=True)

    debug_cfg = getattr(config, "debug", None)
    debug_enable = bool(getattr(debug_cfg, "enable", True)) if debug_cfg is not None else True
    debug_log_every = int(getattr(debug_cfg, "log_every_iters", 20)) if debug_cfg is not None else 20
    debug_first_n = int(getattr(debug_cfg, "dump_first_n_iters", 5)) if debug_cfg is not None else 5
    debug_mirror_dir = str(getattr(debug_cfg, "mirror_dir", "")) if debug_cfg is not None else ""
    debug_mirror_dir = debug_mirror_dir if debug_mirror_dir else None
    if debug_mirror_dir:
        os.makedirs(debug_mirror_dir, exist_ok=True)

    _log_info(f"[act_policy] Hydra output dir: {colored(local_run_output_dir, 'green')}")
    if debug_mirror_dir:
        _log_info(f"[act_policy] Debug mirror dir: {colored(debug_mirror_dir, 'green')}")

    with open(os.path.join(local_run_output_dir, "config_resolved.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))

    # ---- W&B logger ----
    wandb_logger = WandBLogger(
        config=OmegaConf.to_container(config.wandb, resolve=True),
        hyperparameters=OmegaConf.to_container(config, resolve=True),
    )
    try:
        wandb_logger.run.define_metric("train_interation/*", step_metric="iteration_step")
        wandb_logger.run.define_metric("train_epoch/*", step_metric="epoch_step")
        wandb_logger.run.define_metric("validation/*", step_metric="epoch_step")
    except Exception:
        pass

    benchmark_cfg = config.benchmark
    loss_func = instantiate(benchmark_cfg.loss_func, _partial_=True)

    task_name = getattr(config, "task_name", None)
    camera_name = getattr(config, "camera_name", None)
    image_size = int(getattr(config, "image_size", 224))

    # ---- dataset/dataloader ----
    train_base = _instantiate_base_dataset(
        benchmark_cfg.dataset_instantiate_config.base_dataset,
        data_root=config.dataset_dir,
        split="train",
        task_name=task_name,
        camera_name=camera_name,
        image_size=image_size,
    )
    train_dataset = instantiate(benchmark_cfg.dataset_instantiate_config, base_dataset=train_base)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config.dataloader.batch_size),
        shuffle=bool(getattr(config.dataloader, "shuffle", True)),
        num_workers=int(getattr(config.dataloader, "num_workers", 0)),
        drop_last=bool(getattr(config.dataloader, "drop_last", False)),
        pin_memory=bool(getattr(config.dataloader, "pin_memory", True)),
    )

    # ---- infer dims BEFORE model instantiate ----
    sample_batch = next(iter(train_loader))
    _images0, _pc0, _rs0, _raw0, _act0, _txt0, _pad0 = _unpack_batch(sample_batch)

    robot_state_dim = int(_rs0.shape[-1])
    action_dim = int(_act0.shape[-1])

    _log_info(f"[act_policy] inferred robot_state_dim={robot_state_dim}, action_dim={action_dim}")
    _log_info(f"[act_policy] sample shapes: images={tuple(_images0.shape)} pc={tuple(_pc0.shape)} rs={tuple(_rs0.shape)} actions={tuple(_act0.shape)}")
    if _pad0 is not None:
        _log_info(f"[act_policy] sample is_pad shape={tuple(_pad0.shape)} pad_ratio={float(_pad0.float().mean().item()):.4f}")

    with open_dict(config):
        config.agent.instantiate_config.robot_state_dim = robot_state_dim
        config.agent.instantiate_config.action_dim = action_dim

    # ---- build model ----
    model = instantiate(config.agent.instantiate_config).to(config.device)
    try:
        log_params_to_file(model, os.path.join(local_run_output_dir, "model_params.txt"))
        if debug_mirror_dir:
            log_params_to_file(model, os.path.join(debug_mirror_dir, "model_params.txt"))
    except Exception:
        pass

    # ---- validation loader ----
    valid_loader = None
    try:
        valid_base = _instantiate_base_dataset(
            benchmark_cfg.dataset_instantiate_config.base_dataset,
            data_root=config.dataset_dir,
            split="validation",
            task_name=task_name,
            camera_name=camera_name,
            image_size=image_size,
        )
        valid_dataset = instantiate(benchmark_cfg.dataset_instantiate_config, base_dataset=valid_base)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=int(config.dataloader.batch_size),
            shuffle=False,
            num_workers=int(getattr(config.dataloader, "num_workers", 0)),
            drop_last=False,
            pin_memory=bool(getattr(config.dataloader, "pin_memory", True)),
        )
        _log_info(f"[act_policy] validation loader ready: n={len(valid_dataset)} bs={int(config.dataloader.batch_size)}")
    except Exception as e:
        valid_loader = None
        _log_warn(f"[act_policy] validation disabled (cannot build loader): {repr(e)}")

    # ---- optim/sched ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.learning_rate),
        weight_decay=float(getattr(config.train, "weight_decay", 0.0)),
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=int(config.train.num_epochs), eta_min=float(getattr(config.train, "eta_min", 0.0))
    # )

    scheduler = None

    kl_weight = float(getattr(config.train, "kl_weight", 0.0))

    # validation frequency controls
    eval_cfg = getattr(config, "evaluation", None)
    val_freq = int(getattr(eval_cfg, "validation_frequency_epochs", 1)) if eval_cfg is not None else 1
    val_max_batches = getattr(eval_cfg, "validation_max_batches", None) if eval_cfg is not None else None
    val_max_batches = int(val_max_batches) if val_max_batches is not None else None

    debug_jsonl = os.path.join(local_run_output_dir, "debug_steps.jsonl")
    debug_epoch_jsonl = os.path.join(local_run_output_dir, "debug_epochs.jsonl")

    global_iter = 0
    t0 = time.time()

    # (NEW) best tracking
    best_score = float("inf")
    best_tag = None

    _log_info(
        f"[act_policy] start train: epochs={int(config.train.num_epochs)} "
        f"bs={int(config.dataloader.batch_size)} lr={float(config.train.learning_rate)} kl={kl_weight} "
        f"val_freq={val_freq} val_max_batches={val_max_batches}"
    )

    try:
        for cur_epoch in range(int(config.train.num_epochs)):
            model.train()
            loss_train = AverageMeter()

            for cur_iter, batch in enumerate(train_loader):
                global_iter += 1

                images, point_clouds, robot_states, raw_states, actions, texts, is_pad = _unpack_batch(batch)

                images = images.to(config.device, non_blocking=True)
                point_clouds = point_clouds.to(config.device, non_blocking=True)
                robot_states = robot_states.to(config.device, non_blocking=True)
                actions = actions.to(config.device, non_blocking=True)
                if is_pad is not None:
                    is_pad = is_pad.to(config.device, non_blocking=True)

                preds = _maybe_forward_train(model, images, point_clouds, robot_states, texts, actions, is_pad)

                # loss (compat)
                if is_pad is not None:
                    try:
                        loss_result = loss_func(preds, actions, is_pad=is_pad, kl_weight=kl_weight)
                    except TypeError:
                        try:
                            loss_result = loss_func(preds, actions, is_pad=is_pad)
                        except TypeError:
                            loss_result = loss_func(preds, actions)
                else:
                    try:
                        loss_result = loss_func(preds, actions, kl_weight=kl_weight)
                    except TypeError:
                        loss_result = loss_func(preds, actions)

                if isinstance(loss_result, tuple):
                    loss = loss_result[0]
                    loss_dict = loss_result[1] if isinstance(loss_result[1], dict) else {}
                else:
                    loss = loss_result
                    loss_dict = {}

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if float(getattr(config.train, "clip_grad_value", 0.0)) > 0.0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), float(config.train.clip_grad_value))

                optimizer.step()
                loss_train.update(loss.item(), actions.shape[0])

                # wandb step
                iteration_info = {
                    "iteration_step": cur_epoch * len(train_loader) + cur_iter + 1,
                    "train_interation/epoch": cur_epoch,
                    "train_interation/loss": float(loss.item()),
                    "train_interation/learning_rate": float((optimizer.param_groups[0]["lr"])),
                }
                for k, v in loss_dict.items():
                    iteration_info[f"train_interation/{k}"] = _to_item(v)
                wandb_logger.log(iteration_info)

                # debug dump
                if debug_enable and (global_iter <= debug_first_n or (global_iter % debug_log_every == 0)):
                    with torch.no_grad():
                        a_hat = _ensure_chunk(_get_actions_pred(preds).float())
                        actions_ = _ensure_chunk(actions.float())
                        mask_valid = _flatten_valid_mask(is_pad, actions_)
                        pad_ratio = float(is_pad.float().mean().item()) if is_pad is not None else None

                        stats = {
                            "time_sec_from_start": float(time.time() - t0),
                            "epoch": int(cur_epoch),
                            "iter_in_epoch": int(cur_iter),
                            "global_iter": int(global_iter),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "loss": float(loss.item()),
                            "pad_ratio": pad_ratio,
                            "gt_stats": _safe_stats(actions_.float()),
                            "pred_stats": _safe_stats(a_hat.float()),
                            "pred_has_nan": bool(torch.isnan(a_hat).any().item()),
                            "pred_has_inf": bool(torch.isinf(a_hat).any().item()),
                        }
                        stats.update(_masked_mse_mae(a_hat, actions_, mask_valid))
                        stats.update(_cuda_mem())
                        for k, v in loss_dict.items():
                            stats[f"loss_dict/{k}"] = _to_item(v)

                        _maybe_mirror_write_jsonl(debug_jsonl, debug_mirror_dir, stats)

                        msg = (
                            f"[dbg] ep={cur_epoch} it={cur_iter} g={global_iter} "
                            f"loss={stats['loss']:.6g} mse={stats['mse']:.6g} mae={stats['mae']:.6g} "
                        )
                        if pad_ratio is not None:
                            msg += f"pad={pad_ratio:.3f} "
                        if "cuda_alloc_gb" in stats:
                            msg += f"cuda={stats['cuda_alloc_gb']:.2f}/{stats['cuda_reserved_gb']:.2f}GB "
                        if stats["pred_has_nan"] or stats["pred_has_inf"]:
                            msg += "BAD_PRED "
                        _log_info(msg)

            # scheduler.step()
            _log_info(f"[train] epoch={cur_epoch} epoch_loss={loss_train.avg}")

            # epoch summary jsonl
            if debug_enable:
                _maybe_mirror_write_jsonl(
                    debug_epoch_jsonl,
                    debug_mirror_dir,
                    {"epoch": int(cur_epoch), "train_epoch_loss": float(loss_train.avg), "time_sec_from_start": float(time.time() - t0)},
                )

            # (NEW) always save last
            last_path = os.path.join(local_run_output_dir, "last_model.pth")
            _save_checkpoint(
                last_path, model, optimizer, scheduler,
                epoch=cur_epoch, global_iter=global_iter,
                extra={"train_epoch_loss": float(loss_train.avg)}
            )
            if debug_mirror_dir:
                _save_checkpoint(
                    os.path.join(debug_mirror_dir, "last_model.pth"),
                    model, optimizer, scheduler,
                    epoch=cur_epoch, global_iter=global_iter,
                    extra={"train_epoch_loss": float(loss_train.avg)}
                )

            # VALIDATION
            do_val = (valid_loader is not None) and (val_freq > 0) and ((cur_epoch % val_freq) == 0)
            if do_val:
                try:
                    v = validate_one_epoch(
                        model=model,
                        valid_loader=valid_loader,
                        loss_func=loss_func,
                        device=config.device,
                        kl_weight=kl_weight,
                        max_batches=val_max_batches,
                        log_horizon_curve=True,
                    )

                    epoch_info = {
                        "epoch_step": cur_epoch + 1,
                        "train_epoch/loss": float(loss_train.avg),
                        "validation/loss": float(v["val_loss"]),
                        "validation/tf_mse": float(v["val_tf_mse"]),
                        "validation/tf_mae": float(v["val_tf_mae"]),
                        "validation/open_loop_mse": float(v["val_open_loop_mse"]),
                        "validation/open_loop_mae": float(v["val_open_loop_mae"]),
                    }
                    if v.get("val_pad_mean", None) is not None:
                        epoch_info["validation/pad_mean"] = float(v["val_pad_mean"])
                    if v.get("val_pad_loss_corr", None) is not None:
                        epoch_info["validation/pad_loss_corr"] = float(v["val_pad_loss_corr"])

                    if "val_open_loop_horizon_mse_curve" in v:
                        curve = v["val_open_loop_horizon_mse_curve"]
                        for h in range(min(len(curve), 50)):
                            epoch_info[f"validation/open_loop_mse_h{h:02d}"] = float(curve[h])

                    wandb_logger.log(epoch_info)

                    _log_info(
                        f"[val] epoch={cur_epoch} "
                        f"loss={v['val_loss']:.6g} tf(mse/mae)={v['val_tf_mse']:.6g}/{v['val_tf_mae']:.6g} "
                        f"OL(mse/mae)={v['val_open_loop_mse']:.6g}/{v['val_open_loop_mae']:.6g} "
                        f"pad_mean={v.get('val_pad_mean', None)} pad_corr={v.get('val_pad_loss_corr', None)}"
                    )

                    # (NEW) update best: prefer open-loop mse
                    if v.get("val_open_loop_mse", None) is not None:
                        score = float(v["val_open_loop_mse"])
                        tag = "open_loop_mse"
                    else:
                        score = float(v["val_loss"])
                        tag = "val_loss"

                    if score < best_score:
                        best_score = score
                        best_tag = tag
                        best_path = os.path.join(local_run_output_dir, "best_model.pth")
                        _save_checkpoint(
                            best_path, model, optimizer, scheduler,
                            epoch=cur_epoch, global_iter=global_iter,
                            extra={"best_score": best_score, "best_tag": best_tag, **v}
                        )
                        if debug_mirror_dir:
                            _save_checkpoint(
                                os.path.join(debug_mirror_dir, "best_model.pth"),
                                model, optimizer, scheduler,
                                epoch=cur_epoch, global_iter=global_iter,
                                extra={"best_score": best_score, "best_tag": best_tag, **v}
                            )
                        _log_info(f"[ckpt] updated BEST ({best_tag})={best_score:.6g} @ epoch={cur_epoch}")

                except Exception as e:
                    _log_warn(f"[validate] failed (ignored): {repr(e)}")

    except KeyboardInterrupt:
        _log_warn("[interrupt] Caught KeyboardInterrupt. Saving last_model.pth then exiting...")
        try:
            last_path = os.path.join(local_run_output_dir, "last_model.pth")
            _save_checkpoint(
                last_path, model, optimizer, scheduler,
                epoch=-1, global_iter=global_iter,
                extra={"note": "interrupted", "global_iter": int(global_iter)}
            )
            if debug_mirror_dir:
                _save_checkpoint(
                    os.path.join(debug_mirror_dir, "last_model.pth"),
                    model, optimizer, scheduler,
                    epoch=-1, global_iter=global_iter,
                    extra={"note": "interrupted", "global_iter": int(global_iter)}
                )
        except Exception as e:
            _log_warn(f"[interrupt] Failed to save last checkpoint: {repr(e)}")
        raise

    _log_info(f"[DONE] Hydra output dir: {colored(local_run_output_dir, 'green')}")
    if debug_mirror_dir:
        _log_info(f"[DONE] Debug mirror dir: {colored(debug_mirror_dir, 'green')}")


if __name__ == "__main__":
    main()