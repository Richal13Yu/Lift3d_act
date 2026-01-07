# lift3d/dataset/chunk_wrapper.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

try:
    # Only needed when base_dataset is a Hydra config dict
    from hydra.utils import instantiate as hydra_instantiate
except Exception:
    hydra_instantiate = None


@dataclass
class EpisodeStep:
    episode_id: int
    step_id: int


class ChunkDatasetWrapper(Dataset):
    """
    Wrap a *step-based* Lift3D dataset (single-step BC) into a *chunk-supervised* dataset.

    Base dataset item (expected):
        images, point_clouds, robot_states, raw_states, action_t, texts
    (it may return extra fields; we will ignore them)

    Wrapped dataset item:
        images, point_clouds, robot_states, raw_states, actions_chunk, texts, is_pad

    Where:
        actions_chunk: [K, action_dim]
        is_pad:        [K] bool, True means padded (invalid) timestep

    Key feature:
      - Supports fixed episode length (episode_length) so you DON'T need episode_id/step_id inside raw_states.
      - Compatible with Hydra passing (data_dir=..., split=...) into dataset_instantiate_config.

    Extra feature (added here):
      - Optional scaling for position xyz in BOTH robot_states and actions (e.g., meters -> millimeters).
        Set pos_scale=1000.0 to multiply xyz by 1000 during training data loading.
        Default pos_scale=1.0 keeps original behavior.
    """

    def __init__(
        self,
        # Either an already-built Dataset, or a Hydra config dict:
        #   base_dataset:
        #     _target_: lift3d.dataset.MetaWorldDataset
        #     ...
        base_dataset: Union[Dataset, Dict[str, Any]],
        chunk_size: int,
        stride: int = 1,
        pad: bool = True,
        pad_value: float = 0.0,
        # If provided: assume indices are ordered by time and episodes are contiguous blocks
        episode_length: Optional[int] = None,
        # Hydra-compat kwargs (train script often passes these into instantiate(...))
        data_dir: Optional[str] = None,
        split: Optional[str] = None,
        # Optional extra kwargs for base dataset instantiation (if base_dataset is a dict)
        base_dataset_kwargs: Optional[Dict[str, Any]] = None,
        # Advanced / fallback options
        index_to_ep_step: Optional[Callable[[int], EpisodeStep]] = None,
        raw_state_episode_keys: Tuple[str, ...] = ("episode_index", "episode_id", "episode"),
        raw_state_step_keys: Tuple[str, ...] = ("step_index", "frame_index", "t", "step"),
        episode_to_indices: Optional[Dict[int, List[int]]] = None,
        force_torch: bool = True,
        # ---- added: position scaling ----
        pos_scale: float = 1.0,
        pos_dim: int = 3,
    ):
        super().__init__()
        assert chunk_size >= 1, "chunk_size must be >= 1"
        assert stride >= 1, "stride must be >= 1"

        self.K = int(chunk_size)
        self.stride = int(stride)
        self.pad = bool(pad)
        self.pad_value = float(pad_value)
        self.force_torch = bool(force_torch)

        # ---- added: position scaling ----
        self.pos_scale = float(pos_scale)
        self.pos_dim = int(pos_dim)
        if self.pos_dim < 0:
            raise ValueError("pos_dim must be >= 0")

        self.episode_length = int(episode_length) if episode_length is not None else None
        self._raw_state_episode_keys = raw_state_episode_keys
        self._raw_state_step_keys = raw_state_step_keys

        # ---- Build / instantiate base dataset ----
        if isinstance(base_dataset, Dataset):
            self.base: Dataset = base_dataset
        else:
            if hydra_instantiate is None:
                raise ImportError(
                    "hydra is required to instantiate base_dataset from a config dict. "
                    "Either install hydra-core or pass an already constructed Dataset."
                )
            kwargs: Dict[str, Any] = {}
            if base_dataset_kwargs:
                kwargs.update(base_dataset_kwargs)
            # Pass data_dir/split through if provided (these match MetaWorldDataset signature in Lift3D)
            if data_dir is not None:
                kwargs["data_dir"] = data_dir
            if split is not None:
                kwargs["split"] = split
            self.base = hydra_instantiate(base_dataset, **kwargs)

        # ---- Episode mapping ----
        # Fast path: fixed episode length
        if self.episode_length is not None:
            N = len(self.base)
            L = self.episode_length
            if L <= 0:
                raise ValueError("episode_length must be > 0")

            # global index -> (episode_id, step_id)
            def _fixed_index_to_ep_step(i: int) -> EpisodeStep:
                return EpisodeStep(episode_id=i // L, step_id=i % L)

            self.index_to_ep_step = _fixed_index_to_ep_step

            if episode_to_indices is None:
                self.episode_to_indices = self._build_episode_index_map_fixed(N, L)
            else:
                self.episode_to_indices = episode_to_indices

        else:
            # No episode_length: use provided index_to_ep_step, or infer from base dataset/raw_states
            if index_to_ep_step is not None:
                self.index_to_ep_step = index_to_ep_step
            else:
                if hasattr(self.base, "index_to_episode_step") and callable(
                    getattr(self.base, "index_to_episode_step")
                ):
                    self.index_to_ep_step = getattr(self.base, "index_to_episode_step")  # type: ignore
                elif hasattr(self.base, "get_episode_step") and callable(
                    getattr(self.base, "get_episode_step")
                ):
                    self.index_to_ep_step = getattr(self.base, "get_episode_step")  # type: ignore
                else:
                    self.index_to_ep_step = self._infer_from_raw_states

            if episode_to_indices is not None:
                self.episode_to_indices = episode_to_indices
            else:
                if hasattr(self.base, "episode_to_indices"):
                    self.episode_to_indices = getattr(self.base, "episode_to_indices")
                else:
                    self.episode_to_indices = self._build_episode_index_map_scan()

        # Build inverse position map for quick lookup
        self._pos_in_episode: Dict[int, int] = {}
        for ep, idxs in self.episode_to_indices.items():
            for pos, gidx in enumerate(idxs):
                self._pos_in_episode[gidx] = pos

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str) -> Any:
        # Forward unknown attributes to base dataset for convenience
        if name in ("base", "episode_to_indices", "index_to_ep_step"):
            raise AttributeError
        return getattr(self.base, name)

    # ---------- episode/step inference ----------

    def _infer_from_raw_states(self, index: int) -> EpisodeStep:
        """
        Fallback inference by inspecting raw_states (4th field) returned by base[index].
        raw_states must be dict-like or attribute-like.
        """
        item = self.base[index]
        if not isinstance(item, (tuple, list)) or len(item) < 6:
            raise RuntimeError(
                "Base dataset item must be tuple/list with at least 6 fields: "
                "(images, point_clouds, robot_states, raw_states, actions, texts)."
            )

        raw_states = item[3]

        def _get_key(obj: Any, keys: Tuple[str, ...]) -> Optional[int]:
            if isinstance(obj, dict):
                for k in keys:
                    if k in obj:
                        v = obj[k]
                        return int(v) if not torch.is_tensor(v) else int(v.item())
            for k in keys:
                if hasattr(obj, k):
                    v = getattr(obj, k)
                    return int(v) if not torch.is_tensor(v) else int(v.item())
            return None

        ep = _get_key(raw_states, self._raw_state_episode_keys)
        st = _get_key(raw_states, self._raw_state_step_keys)

        if ep is None or st is None:
            raise RuntimeError(
                "Cannot infer (episode_id, step_id) from raw_states. "
                "Provide episode_length=... OR provide index_to_ep_step=... OR provide episode_to_indices=...."
            )

        return EpisodeStep(episode_id=ep, step_id=st)

    def _build_episode_index_map_fixed(self, N: int, L: int) -> Dict[int, List[int]]:
        """
        Fixed-length episodes: episode 0 is indices [0..L-1], episode 1 is [L..2L-1], etc.
        If N is not divisible by L, last episode will be shorter.
        """
        episode_to_indices: Dict[int, List[int]] = {}
        ep = 0
        start = 0
        while start < N:
            end = min(start + L, N)
            episode_to_indices[ep] = list(range(start, end))
            ep += 1
            start = end
        return episode_to_indices

    def _build_episode_index_map_scan(self) -> Dict[int, List[int]]:
        """
        Build episode_id -> indices by scanning all indices and sorting by step_id.
        May be slow because it may call base.__getitem__ if index_to_ep_step depends on raw_states.
        """
        tmp: Dict[int, List[Tuple[int, int]]] = {}  # ep -> [(step_id, global_idx)]
        for gidx in range(len(self.base)):
            epst = self.index_to_ep_step(gidx)
            tmp.setdefault(epst.episode_id, []).append((epst.step_id, gidx))

        episode_to_indices: Dict[int, List[int]] = {}
        for ep, pairs in tmp.items():
            pairs.sort(key=lambda x: x[0])
            episode_to_indices[ep] = [gidx for _, gidx in pairs]
        return episode_to_indices

    # ---------- chunk sampling ----------

    def __getitem__(self, index: int):
        item0 = self.base[index]
        if not isinstance(item0, (tuple, list)) or len(item0) < 6:
            raise RuntimeError(
                "Base dataset item must be tuple/list with at least 6 fields: "
                "(images, point_clouds, robot_states, raw_states, actions, texts)."
            )

        # Keep first 6 fields, ignore any extras
        images, point_clouds, robot_states, raw_states, action_t, texts = item0[:6]

        # Ensure torch tensors for scaling logic
        robot_states = self._to_torch(robot_states).view(-1)
        action_t = self._to_torch(action_t).view(-1)

        # Optional scaling (e.g., meters -> millimeters)
        if self.pos_scale != 1.0 and self.pos_dim > 0:
            rs = robot_states.clone()
            rs[: self.pos_dim] *= self.pos_scale
            robot_states = rs

            at = action_t.clone()
            at[: self.pos_dim] *= self.pos_scale
            action_t = at

        action_dim = int(action_t.numel())

        # Identify episode and position
        epst = self.index_to_ep_step(index)
        ep = epst.episode_id

        if ep not in self.episode_to_indices:
            raise RuntimeError(f"episode_id={ep} not found in episode_to_indices.")
        idxs = self.episode_to_indices[ep]

        if index not in self._pos_in_episode:
            raise RuntimeError(
                f"global index {index} missing in _pos_in_episode. "
                "episode_to_indices mapping is inconsistent with the base dataset."
            )
        pos0 = self._pos_in_episode[index]

        actions_list: List[torch.Tensor] = []
        is_pad_list: List[bool] = []

        for k in range(self.K):
            pos = pos0 + k * self.stride
            if pos < len(idxs):
                gidx_k = idxs[pos]
                item_k = self.base[gidx_k]
                if not isinstance(item_k, (tuple, list)) or len(item_k) < 6:
                    raise RuntimeError("Base dataset item format changed unexpectedly.")

                action_k = self._to_torch(item_k[4]).view(-1)

                if action_k.numel() != action_dim:
                    raise RuntimeError(
                        f"Action dim mismatch inside episode. "
                        f"Expected {action_dim}, got {int(action_k.numel())} at global index {gidx_k}."
                    )

                if self.pos_scale != 1.0 and self.pos_dim > 0:
                    ak = action_k.clone()
                    ak[: self.pos_dim] *= self.pos_scale
                    action_k = ak

                actions_list.append(action_k)
                is_pad_list.append(False)
            else:
                if not self.pad:
                    break
                actions_list.append(
                    torch.full(
                        (action_dim,),
                        self.pad_value,
                        dtype=action_t.dtype,
                        device=action_t.device,
                    )
                )
                is_pad_list.append(True)

        # Always return fixed length K if pad=True
        if self.pad and len(actions_list) != self.K:
            while len(actions_list) < self.K:
                actions_list.append(
                    torch.full(
                        (action_dim,),
                        self.pad_value,
                        dtype=action_t.dtype,
                        device=action_t.device,
                    )
                )
                is_pad_list.append(True)

        actions_chunk = torch.stack(actions_list, dim=0)  # [K,A] (or [<K,A] if pad=False)
        is_pad = torch.tensor(is_pad_list, dtype=torch.bool)

        return images, point_clouds, robot_states, raw_states, actions_chunk, texts, is_pad

    def _to_torch(self, x: Any) -> torch.Tensor:
        if torch.is_tensor(x):
            return x
        if not self.force_torch:
            raise TypeError("force_torch=False but got non-tensor input.")
        return torch.as_tensor(x, dtype=torch.float32)