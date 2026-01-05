# lift3d/envs/dataset_evaluator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from lift3d.envs.evaluator import Evaluator


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


def _default_infer_episode_step(dataset: Dataset, index: int) -> EpisodeStep:
    """
    Best-effort inference from dataset[index][3] (raw_states).

    IMPORTANT:
      - DO NOT use key 't' (torch.Tensor has .t() method).
      - Skip callable attributes.
    """
    item = dataset[index]
    raw_states = item[3]

    ep_keys = ("episode_index", "episode_id", "episode")
    st_keys = ("step_index", "frame_index", "step")  # removed "t"

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
            "or set evaluator episode_length to segment sequentially."
        )
    return EpisodeStep(ep, st)


# --- minimal addition: robustly extract action tensor from policy output ---
def _get_actions_tensor(preds: Any) -> torch.Tensor:
    """
    Supports:
      - Tensor
      - dict with 'actions' / 'a_hat'
      - dataclass/object with attribute '.actions' (e.g., your ActOutput)
    """
    if torch.is_tensor(preds):
        return preds

    if isinstance(preds, dict):
        if "actions" in preds and torch.is_tensor(preds["actions"]):
            return preds["actions"]
        if "a_hat" in preds and torch.is_tensor(preds["a_hat"]):
            return preds["a_hat"]

    if hasattr(preds, "actions"):
        a = getattr(preds, "actions")
        if torch.is_tensor(a):
            return a

    raise TypeError(f"Cannot extract action tensor from model output: type={type(preds)}")


class DatasetEvaluator(Evaluator):
    """
    Offline evaluator: no sim env. Iterate validation dataset.

    Episode ordering priority:
      1) dataset.index_to_episode_step(i) / dataset.get_episode_step(i)
      2) infer from raw_states
      3) sequential segmentation by episode_length
      4) one big episode
    """

    def __init__(
        self,
        dataset_instantiate_config: Any,
        data_dir: str,
        task_name: str,
        split: str = "validation",
        max_episode_length: Optional[int] = None,
        episode_length: Optional[int] = None,
        success_from_error: bool = True,
        success_alpha: float = 5.0,
    ):
        super().__init__()
        from hydra.utils import instantiate

        self.dataset: Dataset = instantiate(
            dataset_instantiate_config,
            data_dir=data_dir,
            split=split,
        )
        self.task_name = task_name
        self.split = split
        self.max_episode_length = max_episode_length
        self.episode_length = episode_length
        self.success_from_error = success_from_error
        self.success_alpha = float(success_alpha)

        self.episode_to_indices: Dict[int, List[int]] = self._build_episode_index_map()

    def _index_to_episode_step(self, index: int) -> EpisodeStep:
        if hasattr(self.dataset, "index_to_episode_step") and callable(getattr(self.dataset, "index_to_episode_step")):
            ep, st = self.dataset.index_to_episode_step(index)  # type: ignore
            return EpisodeStep(int(ep), int(st))
        if hasattr(self.dataset, "get_episode_step") and callable(getattr(self.dataset, "get_episode_step")):
            ep, st = self.dataset.get_episode_step(index)  # type: ignore
            return EpisodeStep(int(ep), int(st))
        return _default_infer_episode_step(self.dataset, index)

    def _build_episode_index_map(self) -> Dict[int, List[int]]:
        tmp: Dict[int, List[Tuple[int, int]]] = {}

        try:
            for idx in range(len(self.dataset)):
                epst = self._index_to_episode_step(idx)
                tmp.setdefault(epst.episode_id, []).append((epst.step_id, idx))

            out: Dict[int, List[int]] = {}
            for ep, pairs in tmp.items():
                pairs.sort(key=lambda x: x[0])
                out[ep] = [i for _, i in pairs]
            return out

        except Exception:
            n = len(self.dataset)
            if self.episode_length is not None and self.episode_length > 0:
                out: Dict[int, List[int]] = {}
                for idx in range(n):
                    ep = idx // int(self.episode_length)
                    out.setdefault(ep, []).append(idx)
                return out

            return {0: list(range(n))}

    @staticmethod
    def _unpack_item(item):
        # support 6-tuple or 7-tuple
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

    @torch.no_grad()
    def evaluate(self, num_episodes: int, policy, verbose: bool = False):
        device = next(policy.parameters()).device

        episode_ids = sorted(list(self.episode_to_indices.keys()))
        if num_episodes is not None:
            episode_ids = episode_ids[: int(num_episodes)]

        total_l1 = 0.0
        total_mse = 0.0
        total_steps = 0
        total_success = 0.0

        for ep in tqdm.tqdm(episode_ids, desc=f"OfflineEval<{self.task_name}:{self.split}>"):
            idxs = self.episode_to_indices[ep]

            if hasattr(policy, "reset_rollout") and callable(getattr(policy, "reset_rollout")):
                policy.reset_rollout()

            steps_this_ep = 0
            sum_l1_ep = 0.0

            for idx in idxs:
                if self.max_episode_length is not None and steps_this_ep >= self.max_episode_length:
                    break

                item = self.dataset[idx]
                images, point_clouds, robot_states, raw_states, actions, texts, is_pad = self._unpack_item(item)

                # batchify to [B=1,...]
                if torch.is_tensor(images) and images.dim() == 3:
                    images = images.unsqueeze(0)
                if torch.is_tensor(point_clouds) and point_clouds.dim() == 2:
                    point_clouds = point_clouds.unsqueeze(0)
                if torch.is_tensor(robot_states) and robot_states.dim() == 1:
                    robot_states = robot_states.unsqueeze(0)

                if isinstance(texts, str):
                    texts_b = [texts]
                elif isinstance(texts, (list, tuple)):
                    texts_b = list(texts)
                else:
                    texts_b = [str(texts)]

                images = images.to(device)
                point_clouds = point_clouds.to(device)
                robot_states = robot_states.to(device)

                # GT action: [A] or [K,A] -> take first step
                a_gt = actions if torch.is_tensor(actions) else torch.as_tensor(actions, dtype=torch.float32)
                if a_gt.dim() == 2:
                    a_gt = a_gt[0]
                a_gt = a_gt.to(device).view(1, -1)

                input_data = {
                    "images": images,
                    "point_clouds": point_clouds,
                    "robot_states": robot_states,
                    "texts": texts_b,
                }

                if hasattr(policy, "act") and callable(getattr(policy, "act")):
                    a_pred = policy.act(**input_data)
                else:
                    a_pred = policy(**input_data)

                # --- minimal fix: support ActOutput/dict/Tensor ---
                a_pred = _get_actions_tensor(a_pred)

                # allow [B,K,A] or [B,A] or [A]
                if a_pred.dim() == 3:
                    a_pred = a_pred[:, 0, :]
                elif a_pred.dim() == 1:
                    a_pred = a_pred.unsqueeze(0)

                a_pred = a_pred.view(1, -1)

                l1 = torch.mean(torch.abs(a_pred - a_gt)).item()
                mse = torch.mean((a_pred - a_gt) ** 2).item()

                sum_l1_ep += l1
                total_l1 += l1
                total_mse += mse
                total_steps += 1
                steps_this_ep += 1

            if self.success_from_error and steps_this_ep > 0:
                avg_l1_ep = sum_l1_ep / steps_this_ep
                total_success += float(np.exp(-self.success_alpha * avg_l1_ep))
            else:
                total_success += 0.0

        if total_steps == 0:
            return 0.0, 0.0

        avg_l1 = total_l1 / total_steps
        avg_success = total_success / max(1, len(episode_ids))
        avg_rewards = -avg_l1
        return avg_success, avg_rewards