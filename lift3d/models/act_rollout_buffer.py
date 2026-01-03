# lift3d/models/act_rollout_buffer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class RolloutConfig:
    mode: str = "replan"  # "replan" or "open_loop"
    chunk_size: int = 50
    temporal_ensemble_coeff: float = 0.0  # 0 -> disable, typical ACT uses ~0.01
    max_history: Optional[int] = None     # default = chunk_size


class ActRolloutBuffer:
    """
    Convert chunk predictions [B,K,A] into step actions [B,A].

    Two modes:

    1) open_loop:
        - predict chunk only when internal queue is empty
        - pop actions sequentially

    2) replan:
        - predict a new chunk every step
        - optional temporal ensembling:
            keep last K chunk predictions; at current time t,
            take chunk predicted i steps ago at its i-th element, weighted by exp(-coeff*i)

    This is designed to be called from model.act(...), while model.forward(...) stays pure.
    """

    def __init__(self, cfg: RolloutConfig):
        self.cfg = cfg
        if self.cfg.max_history is None:
            self.cfg.max_history = self.cfg.chunk_size
        self.reset()

    def reset(self):
        # open_loop queue
        self._queue: Optional[torch.Tensor] = None  # [B, remaining, A]
        self._queue_pos: int = 0

        # replan temporal ensemble
        self._history: List[torch.Tensor] = []  # list of [B,K,A], most recent first

    @torch.no_grad()
    def step_open_loop(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        chunk: [B,K,A]
        return: [B,A]
        """
        if chunk.dim() != 3:
            raise ValueError(f"chunk must be [B,K,A], got {chunk.shape}")

        B, K, A = chunk.shape
        if self._queue is None or (self._queue_pos >= self._queue.shape[1]):
            # refill
            self._queue = chunk
            self._queue_pos = 0

        action = self._queue[:, self._queue_pos, :]
        self._queue_pos += 1
        return action

    @torch.no_grad()
    def step_replan(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        chunk: [B,K,A]
        return: [B,A]
        """
        if chunk.dim() != 3:
            raise ValueError(f"chunk must be [B,K,A], got {chunk.shape}")
        B, K, A = chunk.shape

        coeff = float(self.cfg.temporal_ensemble_coeff)
        if coeff <= 0.0:
            # no ensembling -> execute first action
            return chunk[:, 0, :]

        # push to history (most recent first)
        self._history.insert(0, chunk)

        # keep limited history
        max_h = int(self.cfg.max_history)
        if len(self._history) > max_h:
            self._history = self._history[:max_h]

        # temporal ensemble: use i-th element of chunk predicted i steps ago
        # weights w_i = exp(-coeff * i)
        device = chunk.device
        weights = []
        preds = []

        for i, ch in enumerate(self._history):
            if i >= K:
                break
            preds.append(ch[:, i, :])  # [B,A]
            weights.append(torch.exp(torch.tensor(-coeff * i, device=device)))

        # weighted average
        w = torch.stack(weights, dim=0)  # [H]
        w = w / (w.sum() + 1e-8)
        stacked = torch.stack(preds, dim=0)  # [H,B,A]
        action = (w.view(-1, 1, 1) * stacked).sum(dim=0)  # [B,A]
        return action

    @torch.no_grad()
    def step(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.cfg.mode == "open_loop":
            return self.step_open_loop(chunk)
        elif self.cfg.mode == "replan":
            return self.step_replan(chunk)
        else:
            raise ValueError(f"Unknown rollout mode: {self.cfg.mode}")