import torch
import torch.nn as nn
import numpy as np

from typing import Dict, Union


class Normalizer:
    def __init__(
        self, 
        stats: Dict[str, np.ndarray],
        device: torch.device
    ):
        self._stats = stats
        self._device = device

        # Move stats to torch
        self._stats_torch = {}
        for key in self._stats:
            self._stats_torch[key] = torch.tensor(self._stats[key]).to(device)

    def normalize_numpy(
        self, 
        data: np.ndarray,
        eps : float=1e-3
    ) -> Union[torch.Tensor, np.ndarray]:
        return (data - self._stats["mean"]) / (self._stats["std"] + eps)

    def normalize_torch(
        self, 
        data: torch.Tensor,
        eps : float=1e-3
    ) -> Union[torch.Tensor, np.ndarray]:
        return (data - self._stats_torch["mean"]) / (self._stats_torch["std"] + eps)

    def to(
        self,
        device: torch.device
    ) -> None:
        for key in self._stats_torch:
            self._stats_torch[key] = self._stats_torch[key].to(device)
