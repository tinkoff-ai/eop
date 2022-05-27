from typing import List, cast

import numpy as np

from datasets.dataset import Episode, Transition
from datasets.iterators.base import TransitionIterator


class RoundIterator(TransitionIterator):

    _shuffle: bool
    _indices: np.ndarray
    _index: int

    def __init__(
        self,
        episodes: List[Episode],
        batch_size: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_frames: int = 1,
        shuffle: bool = True,
    ):
        super().__init__(
            episodes=episodes,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            n_frames=n_frames,
        )
        self._shuffle = shuffle
        self._indices = np.arange(len(self._transitions))
        self._index = 0

    def _reset(self) -> None:
        self._indices = np.arange(len(self._transitions))
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._index = 0

    def _next(self) -> Transition:
        transition = self._transitions[cast(int, self._indices[self._index])]
        self._index += 1
        return transition

    def _has_finished(self) -> bool:
        return self._index >= len(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions) // self._real_batch_size