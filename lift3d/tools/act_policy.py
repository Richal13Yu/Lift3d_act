#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lift3d/tools/act_policy.py

ACT-style training script (chunk supervision + optional VAE KL loss),
compatible with Lift3D Hydra config layout.

Expected dataset batch formats:
  - 6-tuple: images, point_clouds, robot_states, raw_states, actions, texts
  - 7-tuple: images, point_clouds, robot_states, raw_states, actions_chunk, texts, is_pad

For ACT chunk training:
  actions_chunk: [B, K, A]
  is_pad:        [B, K] bool (True means padded)

Model forward should accept:
  preds = model(images, point_clouds, robot_states, texts)

And return either:
  - Tensor: [B, K, A] (or [B, A] for single-step)
  - dict with at least:
      "actions" or "a_hat": [B, K, A]
    optionally:
      "mu", "logvar" (for CVAE KL), "is_pad_hat" etc.

Loss function (Hydra) is called as:
  loss_result = loss_func(preds, actions, is_pad=is_pad)
and may return:
  - loss tensor
  - or (loss_tensor, metrics_dict)
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
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
from termcolor import colored

from lift3d.envs import Evaluator
from lift3d.helpers.common import Logger, WandBLogger, set_seed
from lift3d.helpers.pytorch import AverageMeter, log_params_to_file


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor, Any, Optional[torch.Tensor]]:
    """
    Returns:
      images, point_clouds, robot_states, raw_states, actions, texts, is_pad(optional)
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


def _get_actions_pred(preds) -> torch.Tensor:
    """
    Extract action prediction tensor from model output.
    """
    if torch.is_tensor(preds):
        return preds
    if isinstance(preds, dict):
        if "actions" in preds:
            return preds["actions"]
        if "a_hat" in preds:
            return preds["a_hat"]
    raise ValueError(
        "Model output must be a Tensor or a dict containing 'actions' or 'a_hat'. "
        f"Got type={type(preds)} keys={list(preds.keys()) if isinstance(preds, dict) else None}"
    )


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
    Logger.log_info(f'Image size: {colored(config.image_size, "green")}')
    Logger.log_info(
        f'WandB: Project {colored(config.wandb.project, "green")}; '
        f'Group {colored(config.wandb.group, "green")}; '
        f'Name {colored(config.wandb.name, "green")}; '
        f'Notes {colored(config.wandb.notes, "green")}; '
        f'Mode {colored(config.wandb.mode, "green")}'
    )
    Logger.log_info(
        f'Agent: {colored(config.agent.name, color="green")}\n'
        f'{json.dumps(OmegaConf.to_container(config.agent, resolve=True), indent=4)}'
    )
    Logger.log_info(
        f'Benchmark: {colored(config.benchmark.name, color="green")}\n'
        f'{json.dumps(OmegaConf.to_container(config.benchmark, resolve=True), indent=4)}'
    )
    Logger.print_seperator()

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
    wandb_logger.run.define_metric("train_interation/*", step_metric="iteration_step")
    wandb_logger.run.define_metric("train_epoch/*", step_metric="epoch_step")
    wandb_logger.run.define_metric("validation/*", step_metric="epoch_step")

    # ----------------------------
    # Datasets
    # ----------------------------
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
        raise RuntimeError(
            "Train dataset is empty (len==0). "
            "Check dataset_dir, split name, and that your dataset class enumerates indices correctly."
        )
    if len(valid_dataset) == 0:
        Logger.log_warn(
            "Validation dataset is empty (len==0). Validation will run but have no batches."
        )

    # ----------------------------
    # Optional evaluator
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
        except TypeError:
            # Fallback: evaluator doesn't accept dataset args
            evaluator = instantiate(
                config=config.benchmark.evaluator_instantiate_config,
                task_name=config.task_name,
            )

    # ----------------------------
    # DataLoaders
    # ----------------------------
    DataLoaderConstructor = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        shuffle=config.dataloader.shuffle,
        pin_memory=config.dataloader.pin_memory,
        drop_last=config.dataloader.drop_last,
    )
    train_loader = DataLoaderConstructor(train_dataset)
    valid_loader = DataLoaderConstructor(valid_dataset)

    # ----------------------------
    # Infer dims from one batch
    # ----------------------------
    it = iter(train_loader)
    try:
        sample_batch = next(it)
    except StopIteration as e:
        raise RuntimeError(
            "Train DataLoader produced no batches (StopIteration). "
            "Common causes: dataset length is 0, or drop_last=True with len(dataset) < batch_size."
        ) from e

    _, _, sample_robot_state, _, sample_action, _, _ = _unpack_batch(sample_batch)

    robot_state_dim = int(sample_robot_state.size(-1))
    # works for [B,A] or [B,K,A]
    action_dim = int(sample_action.size(-1))

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

    # ----------------------------
    # Optimizer / Scheduler
    # ----------------------------
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.train.learning_rate,
    )
    local_run_output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    log_params_to_file(model, local_run_output_dir / "train_model_params.txt", True)
    log_params_to_file(model, local_run_output_dir / "train_model_params_freeze.txt", False)

    scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate(
        config=config.train.scheduler_instantiate_config,
        optimizer=optimizer,
    )

    # ----------------------------
    # Training loop
    # ----------------------------
    best_success = -1.0
    best_val_loss = float("inf")
    max_success, max_rewards = 0.0, 0.0

    for cur_epoch in range(config.train.num_epochs):
        epoch_logging_info: Dict[str, Any] = {"epoch_step": cur_epoch + 1}

        # ---- train ----
        model.train()
        loss_train = AverageMeter()

        for cur_iter, batch in enumerate(train_loader):
            iteration_info: Dict[str, Any] = {}

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

            # Loss call (prefer passing is_pad when available)
            if is_pad is not None:
                try:
                    loss_result = call(config.benchmark.loss_func, preds, actions, is_pad=is_pad)
                except TypeError:
                    loss_result = call(config.benchmark.loss_func, preds, actions)
            else:
                loss_result = call(config.benchmark.loss_func, preds, actions)

            if isinstance(loss_result, tuple):
                loss = loss_result[0]
                loss_dict = loss_result[1]
                for k, v in loss_dict.items():
                    iteration_info[f"train_interation/{k}"] = _to_item(v)
            else:
                loss = loss_result

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if config.train.clip_grad_value > 0.0:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.train.clip_grad_value)

            optimizer.step()
            loss_train.update(loss.item())

            iteration_info.update(
                {
                    "iteration_step": cur_epoch * len(train_loader) + cur_iter + 1,
                    "train_interation/epoch": cur_epoch,
                    "train_interation/loss": loss.item(),
                    "train_interation/learning_rate": scheduler.get_last_lr()[0],
                }
            )
            wandb_logger.log(iteration_info)

        scheduler.step()
        epoch_logging_info["train_epoch/epoch_loss"] = loss_train.avg
        Logger.log_info(f"[train] epoch={cur_epoch}, loss={loss_train.avg}")

        # ---- validation ----
        periodic_validation = (cur_epoch + 1 > config.evaluation.num_skip_epochs) and (
            (cur_epoch + 1) % config.evaluation.validation_frequency_epochs == 0
        )
        last_epoch = (cur_epoch + 1) == config.train.num_epochs

        if periodic_validation or last_epoch:
            model.eval()
            loss_val = AverageMeter()

            for batch in valid_loader:
                images, point_clouds, robot_states, raw_states, actions, texts, is_pad = _unpack_batch(batch)

                images = images.to(config.device)
                point_clouds = point_clouds.to(config.device)
                robot_states = robot_states.to(config.device)
                actions = actions.to(config.device, non_blocking=True)
                if is_pad is not None:
                    is_pad = is_pad.to(config.device, non_blocking=True)

                with torch.no_grad():
                    preds = model(
                        images, point_clouds, robot_states, texts,
                        actions=actions,
                        is_pad=is_pad,
                    )

                if is_pad is not None:
                    try:
                        loss_result = call(config.benchmark.loss_func, preds, actions, is_pad=is_pad)
                    except TypeError:
                        loss_result = call(config.benchmark.loss_func, preds, actions)
                else:
                    loss_result = call(config.benchmark.loss_func, preds, actions)

                if isinstance(loss_result, tuple):
                    loss_val.update(loss_result[0].item(), actions.shape[0])
                    loss_dict = loss_result[1]
                    for k, v in loss_dict.items():
                        epoch_logging_info[f"validation/{k}"] = _to_item(v)
                else:
                    loss_val.update(loss_result.item(), actions.shape[0])

            epoch_logging_info.update(
                {
                    "validation/epoch": cur_epoch,
                    "validation/loss": loss_val.avg,
                }
            )

            # ---- rollout eval (optional) ----
            if evaluator is not None:
                avg_success, avg_rewards = evaluator.evaluate(
                    config.evaluation.validation_trajs_num, model
                )
                max_success = max(max_success, avg_success)
                max_rewards = max(max_rewards, avg_rewards)

                epoch_logging_info.update(
                    {
                        "validation/success": avg_success,
                        "validation/rewards": avg_rewards,
                        "validation/max_success": max_success,
                        "validation/max_rewards": max_rewards,
                    }
                )

                # video if available
                if getattr(evaluator, "env", None) is not None and hasattr(evaluator.env, "get_frames"):
                    try:
                        epoch_logging_info["validation/video_steps"] = wandb.Video(
                            evaluator.env.get_frames().transpose(0, 3, 1, 2), fps=30
                        )
                    except Exception:
                        pass

                evaluator.callback(epoch_logging_info)

                Logger.log_info(
                    f"[validation] epoch={cur_epoch}, "
                    f"val_loss={loss_val.avg}, success={avg_success}, rewards={avg_rewards}"
                )

                # save best by success
                if config.evaluation.save_best_model and avg_success > best_success:
                    best_success = avg_success
                    model_path = os.path.join(local_run_output_dir, "best_model.pth")
                    torch.save(model.state_dict(), model_path)
                    Logger.log_info(f"Save best model (by success) to {colored(model_path, 'red')}")
                    with open(os.path.join(local_run_output_dir, "best_model.json"), "w") as f:
                        json.dump(
                            {
                                "epoch": cur_epoch,
                                "val_loss": loss_val.avg,
                                "avg_success": avg_success,
                                "avg_rewards": avg_rewards,
                                "criterion": "max_success",
                            },
                            f,
                            indent=4,
                        )

            else:
                # offline-only: save best by lowest val loss
                epoch_logging_info.update(
                    {
                        "validation/success": 0.0,
                        "validation/rewards": 0.0,
                        "validation/max_success": 0.0,
                        "validation/max_rewards": 0.0,
                    }
                )
                Logger.log_info(f"[validation/offline] epoch={cur_epoch}, val_loss={loss_val.avg}")

                if config.evaluation.save_best_model and loss_val.avg < best_val_loss:
                    best_val_loss = loss_val.avg
                    model_path = os.path.join(local_run_output_dir, "best_model.pth")
                    torch.save(model.state_dict(), model_path)
                    Logger.log_info(f"Save best model (by val loss) to {colored(model_path, 'red')}")
                    with open(os.path.join(local_run_output_dir, "best_model.json"), "w") as f:
                        json.dump(
                            {
                                "epoch": cur_epoch,
                                "val_loss": loss_val.avg,
                                "criterion": "min_val_loss",
                            },
                            f,
                            indent=4,
                        )

        wandb_logger.log(epoch_logging_info)

    Logger.log_ok("Training Finished!")


if __name__ == "__main__":
    main()