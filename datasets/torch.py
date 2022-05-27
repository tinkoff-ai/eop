import numpy as np
import torch

from typing import Optional
from datasets.dataset import TransitionMiniBatch, MDPDataset, Episode, Transition
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from typing import List


def _convert_to_torch(array: np.ndarray, device: str) -> Tensor:
    dtype = torch.uint8 if array.dtype == np.uint8 else torch.float32
    tensor = torch.tensor(data=array, dtype=dtype, device=device)
    return tensor.float()


class TorchMiniBatch:
    _observations: Tensor
    _actions: Tensor
    _rewards: Tensor
    _next_observations: Tensor
    _next_actions: Tensor
    _next_rewards: Tensor
    _terminals: Tensor
    _masks: Optional[Tensor]
    _n_steps: Tensor
    _device: str

    def __init__(
        self,
        batch: TransitionMiniBatch,
        device: str,
    ):
        # convert numpy array to torch tensor
        observations = _convert_to_torch(batch.observations, device)
        actions = _convert_to_torch(batch.actions, device)
        rewards = _convert_to_torch(batch.rewards, device)
        next_observations = _convert_to_torch(batch.next_observations, device)
        next_actions = _convert_to_torch(batch.next_actions, device)
        next_rewards = _convert_to_torch(batch.next_rewards, device)
        terminals = _convert_to_torch(batch.terminals, device)
        masks: Optional[Tensor]

        if batch.masks is None:
            masks = None
        else:
            masks = _convert_to_torch(batch.masks, device)
        n_steps = _convert_to_torch(batch.n_steps, device)

        self._observations = observations
        self._actions = actions
        self._rewards = rewards
        self._next_observations = next_observations
        self._next_actions = next_actions
        self._next_rewards = next_rewards
        self._terminals = terminals
        self._masks = masks
        self._n_steps = n_steps
        self._device = device

    @property
    def observations(self) -> Tensor:
        return self._observations

    @property
    def actions(self) -> Tensor:
        return self._actions

    @property
    def rewards(self) -> Tensor:
        return self._rewards

    @property
    def next_observations(self) -> Tensor:
        return self._next_observations

    @property
    def next_actions(self) -> Tensor:
        return self._next_actions

    @property
    def next_rewards(self) -> Tensor:
        return self._next_rewards

    @property
    def terminals(self) -> Tensor:
        return self._terminals

    @property
    def masks(self) -> Optional[Tensor]:
        return self._masks

    @property
    def n_steps(self) -> Tensor:
        return self._n_steps

    @property
    def device(self) -> str:
        return self._device


class InfiniteMDPIterator:
    def __init__(self, mdp_dataset: MDPDataset):
        self._mdp_dataset = mdp_dataset

    def __iter__(self):
        return self

    def __next__(self):
        episode_ind = np.random.randint(0, len(self._mdp_dataset.episodes))
        trans_ind   = np.random.randint(0, len(self._mdp_dataset.episodes[episode_ind]))
        transition  = self._mdp_dataset.episodes[episode_ind].transitions[trans_ind]

        return {
            "observation": transition.observation.astype(dtype=np.float32),
            "next_observation": transition.next_observation.astype(dtype=np.float32),
            "action": transition.action.astype(dtype=np.float32),
            "terminal": np.array(transition.terminal, dtype=np.float32),
            "reward": np.array(transition.reward, dtype=np.float32)
        }


class InfiniteMDPTorchDataset(IterableDataset):
    def __init__(self, mdp_dataset: MDPDataset):
        self._mdp_dataset = mdp_dataset
        # warn: this thing apparently does a copy
        # self._transitions: List[Transition] = []
        # for episode in mdp_dataset.episodes:
        #     self._transitions += episode.transitions

    def __iter__(self):
        return iter(InfiniteMDPIterator(self._mdp_dataset))


class FiniteMDPTorchDataset(Dataset):
    def __init__(
        self, 
        mdp_dataset: MDPDataset,
        s0_only: bool = False
    ):
        self._transitions: List[Transition] = []
        for episode in mdp_dataset.episodes:
            if s0_only:
                self._transitions.append(episode.transitions[0])
            else:
                self._transitions += episode.transitions

    def __getitem__(self, index):
        transition = self._transitions[index]

        return {
            "observation": transition.observation.astype(dtype=np.float32),
            "next_observation": transition.next_observation.astype(dtype=np.float32),
            "action": transition.action.astype(dtype=np.float32),
            "terminal": np.array(transition.terminal, dtype=np.float32),
            "reward": np.array(transition.reward, dtype=np.float32)
        }

    def __len__(self):
        return len(self._transitions)
