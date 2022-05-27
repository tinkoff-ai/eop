from typing import List, cast

import numpy as np

from datasets.dataset import Episode, Transition
from datasets.iterators.base import TransitionIterator


class RandomIterator(TransitionIterator):

    _n_steps_per_epoch: int

    def __init__(
        self,
        episodes: List[Episode],
        n_steps_per_epoch: int,
        batch_size: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_frames: int = 1,
    ):
        super().__init__(
            episodes=episodes,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            n_frames=n_frames,
        )
        self._n_steps_per_epoch = n_steps_per_epoch

    def _reset(self) -> None:
        pass

    def _next(self) -> Transition:
        index = cast(int, np.random.randint(len(self._transitions)))
        transition = self._transitions[index]
        return transition

    def _has_finished(self) -> bool:
        return self._count >= self._n_steps_per_epoch

    def __len__(self) -> int:
        return self._n_steps_per_epoch