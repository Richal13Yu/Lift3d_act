#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lift3d/tools/act_policy_normal.py

ACT-style training script with Auto-Normalization.
Fixes convergence issues for small-scale surgical data by automatically 
computing Z-Score statistics (mean/std) and injecting them into the model.
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
from typing import Optional, Tuple, Any, Dict

import hydra
import torch
import wandb
import numpy as np
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from lift3d.envs.evaluator import Evaluator
from lift3d.helpers.common import Logger, WandBLogger, set_seed
from lift3d.helpers.pytorch import AverageMeter, log_params_to_file


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor, Any, Optional[torch.Tensor]]:
    """
    Robust batch unpacking handling both 6-tuple (old) and 7-tuple (with is_pad).
    Returns:
      images, point_clouds, robot_states, raw_states, actions, texts, is_pad
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


def _to_item(v):
    if torch.is_tensor(v):
        return v.item()
    return v


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(config):
    # ----------------------------
    # Log config
    # ----------------------------
    Logger.log_info(
        f'Running {colored(pathlib.Path(__file__).absolute(), "red")} with following config:'
    )
    Logger.log_info(f'Task: {colored(config.task_name, "green")}')
    Logger.log_info(f'Dataset directory: {colored(config.dataset_dir, "green")}')
    
    # ----------------------------
    # Seed
    # ----------------------------
    set_seed(config.seed)

    # ----------------------------
    # W&B logger
    # ----------------------------
    wandb_logger = WandBLogger(
        config=config.wandb,
        hyperparameters=OmegaConf.to_container(config, resolve=True),
    )
    # Define metrics to prevent UI flickering
    wandb_logger.run.define_metric("train_iteration/*", step_metric="iteration_step")
    wandb_logger.run.define_metric("train_epoch/*", step_metric="epoch_step")
    wandb_logger.run.define_metric("validation/*", step_metric="epoch_step")

    # ----------------------------
    # Datasets
    # ----------------------------
    # Ensure pos_scale is passed as 1.0 ideally, letting Auto-Norm handle scaling
    train_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        data_dir=config.dataset_dir,
        split="train",
    )
    valid_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        data_dir=config.dataset_dir,
        split="validation",
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty.")

    # ----------------------------
    # DataLoaders Construction
    # ----------------------------
    DataLoaderConstructor = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        drop_last=config.dataloader.drop_last,
    )
    
    # ----------------------------
    # AUTO-NORMALIZATION: Compute Statistics
    # ----------------------------
    # This block scans the dataset to compute Mean and Std for Z-Score Normalization.
    # Essential for surgical data with small motion ranges.
    Logger.print_seperator()
    Logger.log_info(colored(" [Auto-Norm] Calculating dataset statistics...", "cyan"))
    
    # Use a temporary loader (no shuffle) to scan data
    stats_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=config.dataloader.num_workers
    )

    all_actions = []
    all_qpos = []
    max_samples_for_stats = 10000 # Limit samples to avoid OOM
    collected_samples = 0

    for batch in tqdm(stats_loader, desc="Computing Stats"):
        _, _, robot_states, _, actions, _, _ = _unpack_batch(batch)
        all_actions.append(actions)
        all_qpos.append(robot_states)
        
        collected_samples += actions.shape[0]
        if collected_samples >= max_samples_for_stats:
            break
    
    # Concatenate and Flatten
    # actions: [N, T, D] -> [N*T, D]
    all_actions = torch.cat(all_actions, dim=0)
    flat_actions = all_actions.reshape(-1, all_actions.shape[-1])
    
    # qpos: [N, D] -> [N, D]
    all_qpos = torch.cat(all_qpos, dim=0)
    flat_qpos = all_qpos.reshape(-1, all_qpos.shape[-1])
    
    # Compute Stats
    action_mean = flat_actions.mean(dim=0)
    action_std = flat_actions.std(dim=0)
    qpos_mean = flat_qpos.mean(dim=0)
    qpos_std = flat_qpos.std(dim=0)
    
    # Clip small std to prevent division by zero or explosion
    action_std = torch.clip(action_std, min=1e-5)
    qpos_std = torch.clip(qpos_std, min=1e-5)

    Logger.log_info(f" [Auto-Norm] Action Mean: {action_mean[:3].tolist()}...")
    Logger.log_info(f" [Auto-Norm] Action Std:  {action_std[:3].tolist()}...")
    Logger.print_seperator()

    # ----------------------------
    # Final DataLoaders
    # ----------------------------
    train_loader = DataLoaderConstructor(train_dataset, shuffle=config.dataloader.shuffle)
    valid_loader = DataLoaderConstructor(valid_dataset, shuffle=False)

    # Infer dims
    robot_state_dim = int(qpos_mean.shape[0])
    action_dim = int(action_mean.shape[0])
    Logger.log_info(f'Robot state dim: {colored(robot_state_dim, "red")}')
    Logger.log_info(f'Action dim: {colored(action_dim, "red")}')

    # ----------------------------
    # Model
    # ----------------------------
    model = instantiate(
        config=config.agent.instantiate_config,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
    ).to(config.device)

    # --- INJECT STATS INTO MODEL ---
    # This requires the model class (ACTPolicy) to have a set_norm_stats method.
    if hasattr(model, "set_norm_stats"):
        model.set_norm_stats(
            action_mean.to(config.device), 
            action_std.to(config.device), 
            qpos_mean.to(config.device), 
            qpos_std.to(config.device)
        )
        Logger.log_info(colored(" [Auto-Norm] Statistics injected into model successfully!", "green"))
    else:
        Logger.log_warn(colored(" [Warning] Model missing 'set_norm_stats'. Auto-Norm skipped.", "yellow"))

    # ----------------------------
    # Optimizer / Scheduler
    # ----------------------------
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=OmegaConf.select(config, "train.weight_decay", default=1e-4) # Added default weight decay
    )
    
    # Save params for reference
    local_run_output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    log_params_to_file(model, local_run_output_dir / "train_model_params.txt", True)

    scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate(
        config=config.train.scheduler_instantiate_config,
        optimizer=optimizer,
    )

    # ----------------------------
    # KL Weight Config
    # ----------------------------
    kl_weight = OmegaConf.select(config, "train.kl_weight", default=None)
    if kl_weight is None:
        kl_weight = OmegaConf.select(config, "agent.kl_weight", default=None)
    if kl_weight is None:
        kl_weight = OmegaConf.select(config, "agent.instantiate_config.kl_weight", default=10.0)
    kl_weight = float(kl_weight)
    Logger.log_info(f"KL weight: {colored(kl_weight, 'red')}")

    # ----------------------------
    # Evaluator
    # ----------------------------
    evaluator: Optional[Evaluator] = None
    if getattr(config.benchmark, "evaluator_instantiate_config", None) is not None:
        try:
            evaluator = instantiate(
                config=config.benchmark.evaluator_instantiate_config,
                task_name=config.task_name,
                data_dir=config.dataset_dir,
                dataset_instantiate_config=config.benchmark.dataset_instantiate_config,
            )
        except Exception as e:
            Logger.log_warn(f"Failed to instantiate evaluator: {e}")

    # ----------------------------
    # Training Loop
    # ----------------------------
    best_success = -1.0
    best_val_loss = float("inf")
    
    # Get clip value from config or default to safe value
    clip_grad_value = OmegaConf.select(config, "train.clip_grad_value", default=10.0)

    for cur_epoch in range(config.train.num_epochs):
        epoch_logging_info: Dict[str, Any] = {"epoch_step": cur_epoch + 1}

        # ---- Train ----
        model.train()
        loss_train = AverageMeter()

        for cur_iter, batch in enumerate(train_loader):
            iteration_info: Dict[str, Any] = {}

            # Unpack
            images, point_clouds, robot_states, raw_states, actions, texts, is_pad = _unpack_batch(batch)

            # Move to device
            images = images.to(config.device)
            point_clouds = point_clouds.to(config.device)
            robot_states = robot_states.to(config.device)
            actions = actions.to(config.device, non_blocking=True)
            if is_pad is not None:
                is_pad = is_pad.to(config.device, non_blocking=True)

            # Forward
            # Note: Model should handle Normalization internally using injected stats
            actions_norm = (actions - model.action_mean) / model.action_std
            preds = model(
                images, point_clouds, robot_states, texts,
                actions=actions,
                is_pad=is_pad,
            )

            # Loss Calculation
            # Flexible calling convention for different loss functions
            kwargs = {'kl_weight': kl_weight}
            if is_pad is not None:
                kwargs['is_pad'] = is_pad
            
            # Helper to call loss func safely
            try:
                loss_result = call(config.benchmark.loss_func, preds, actions_norm, **kwargs)
            except TypeError:
                # Fallback if loss func doesn't accept all kwargs
                if 'kl_weight' in kwargs: del kwargs['kl_weight']
                loss_result = call(config.benchmark.loss_func, preds, actions_norm, **kwargs)

            # Handle loss return types
            if isinstance(loss_result, tuple):
                loss, loss_dict = loss_result
                if isinstance(loss_dict, dict):
                    for k, v in loss_dict.items():
                        iteration_info[f"train_iteration/{k}"] = _to_item(v)
            else:
                loss = loss_result

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient Clipping
            if clip_grad_value > 0.0:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)

            optimizer.step()
            loss_train.update(loss.item())

            # Log Iteration
            iteration_info.update({
                "iteration_step": cur_epoch * len(train_loader) + cur_iter + 1,
                "train_iteration/epoch": cur_epoch,
                "train_iteration/loss": loss.item(),
                "train_iteration/learning_rate": scheduler.get_last_lr()[0],
            })
            wandb_logger.log(iteration_info)

        scheduler.step()
        epoch_logging_info["train_epoch/epoch_loss"] = loss_train.avg
        Logger.log_info(f"[train] epoch={cur_epoch}, loss={loss_train.avg:.6f}")

        # ---- Validation ----
        # Validate periodically or on last epoch
        periodic_validation = (cur_epoch + 1 > config.evaluation.num_skip_epochs) and (
            (cur_epoch + 1) % config.evaluation.validation_frequency_epochs == 0
        )
        last_epoch = (cur_epoch + 1) == config.train.num_epochs

        if periodic_validation or last_epoch:
            model.eval()
            loss_val = AverageMeter()

            with torch.no_grad():
                for batch in valid_loader:
                    images, point_clouds, robot_states, raw_states, actions, texts, is_pad = _unpack_batch(batch)

                    images = images.to(config.device)
                    point_clouds = point_clouds.to(config.device)
                    robot_states = robot_states.to(config.device)
                    actions = actions.to(config.device, non_blocking=True)
                    if is_pad is not None:
                        is_pad = is_pad.to(config.device, non_blocking=True)

                    preds = model(
                        images, point_clouds, robot_states, texts,
                        actions=actions,
                        is_pad=is_pad,
                    )

                    # Validation Loss
                    kwargs = {'kl_weight': kl_weight}
                    if is_pad is not None: kwargs['is_pad'] = is_pad
                    
                    try:
                        loss_result = call(config.benchmark.loss_func, preds, actions, **kwargs)
                    except TypeError:
                         if 'kl_weight' in kwargs: del kwargs['kl_weight']
                         loss_result = call(config.benchmark.loss_func, preds, actions, **kwargs)

                    if isinstance(loss_result, tuple):
                        val_loss_val = loss_result[0]
                    else:
                        val_loss_val = loss_result
                    
                    loss_val.update(val_loss_val.item(), actions.shape[0])

            epoch_logging_info.update({
                "validation/epoch": cur_epoch,
                "validation/loss": loss_val.avg,
            })
            Logger.log_info(f"[validation] epoch={cur_epoch}, val_loss={loss_val.avg:.6f}")

            # ---- Evaluator (Rollout) ----
            avg_success = 0.0
            avg_rewards = 0.0
            
            if evaluator is not None:
                avg_success, avg_rewards = evaluator.evaluate(
                    config.evaluation.validation_trajs_num, model
                )
                epoch_logging_info.update({
                    "validation/success": avg_success,
                    "validation/rewards": avg_rewards,
                })
                Logger.log_info(f"[evaluator] success={avg_success}, rewards={avg_rewards}")

            # ---- Save Checkpoints ----
            # Save if better success OR better val loss (fallback)
            
            saved = False
            # Criteria 1: Success Rate (if evaluator exists)
            if evaluator is not None and config.evaluation.save_best_model and avg_success > best_success:
                best_success = avg_success
                saved = True
                
            # Criteria 2: Val Loss (if offline or equal success)
            if config.evaluation.save_best_model and loss_val.avg < best_val_loss:
                best_val_loss = loss_val.avg
                # Only save if we haven't saved for success yet, or if success didn't improve
                if evaluator is None: 
                    saved = True

            if saved:
                model_path = os.path.join(local_run_output_dir, "best_model.pth")
                torch.save(model.state_dict(), model_path)
                Logger.log_info(f"Save best model to {colored(model_path, 'red')}")
                
                with open(os.path.join(local_run_output_dir, "best_model.json"), "w") as f:
                    json.dump({
                        "epoch": cur_epoch,
                        "val_loss": loss_val.avg,
                        "avg_success": avg_success,
                    }, f, indent=4)
            
            # Save Last Model
            torch.save(model.state_dict(), os.path.join(local_run_output_dir, "last_model.pth"))

        wandb_logger.log(epoch_logging_info)

    Logger.log_ok("Training Finished!")


if __name__ == "__main__":
    main()