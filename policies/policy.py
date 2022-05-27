from os import stat
from typing import Any, List

from torch.functional import norm
from utils.normalize import Normalizer

import gym
import torch
import torch.nn as nn

from datasets.torch import _convert_to_torch
from nets import TanhGaussianMLPNet, GMMMlpNet
from typing import Optional


class Policy:
    _device: torch.device
    _normalizer: Optional[Normalizer] = None

    def predict_actions(self, obs: List[Any]) -> List[Any]:
        raise NotImplementedError()

    def predict_actions_torch(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        raise NotImplementedError()

    def to(self, device: torch.device) -> None:
        self._device = device

    def set_normalizer(self, normalizer: Optional[Normalizer]) -> None:
        self._normalizer = normalizer


class RandomPolicy(Policy):
    def __init__(self, env: gym.Env):
        self.action_space = env.action_space

    def predict_actions(self, obs: List[Any]) -> List[Any]:
        return [self.action_space.sample() for _ in range(len(obs))]


class ValueBasedPolicy(Policy):
    def __init__(self, value_network: nn.Module) -> None:
        super().__init__()
        self._value_network = value_network

    def predict_actions(self, obs: List[Any]) -> List[Any]:
        with torch.no_grad():
            obs     = _convert_to_torch(array=obs, device=self._device)
            actions = self._predict_actions_torch(obs)

            return actions.cpu().numpy().tolist()

    def predict_actions_torch(self, obs: torch.Tensor) -> torch.Tensor:
        values  = self._value_network.forward(obs)
        actions = torch.argmax(input=values, dim=1)

        return actions

    def save(self, path: str) -> None:
        torch.save(self._value_network.state_dict(), path)

    def to(self, device: torch) -> None:
        self._device = device
        self._value_network.to(self._device)

        if self._normalizer:
            self._normalizer.to(self._device)

    @staticmethod
    def load(path: str, value_network: nn.Module) -> Policy:
        value_network.load_state_dict(torch.load(path, map_location="cpu"))
        
        return ValueBasedPolicy(value_network)


class ActorBasedPolicy(Policy):
    def __init__(
        self, 
        actor_network: nn.Module,
        normalizer: Optional[Normalizer]
    ) -> None:
        super().__init__()
        self._actor_network = actor_network
        self.set_normalizer(normalizer)

    def predict_actions(self, obs: List[Any]) -> List[Any]:
        obs     = _convert_to_torch(array=obs, device=self._device)
        actions = self.predict_actions_torch(obs)

        return actions.cpu().numpy()

    def predict_actions_torch(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Normalize if needed
            if self._normalizer:
                obs = self._normalizer.normalize_torch(obs)
            actions  = self._actor_network.forward(obs)

            return actions

    def save(self, path: str) -> None:
        torch.save({
            "actor": self._actor_network.state_dict(),
            "normalizer": self._normalizer
        }, path)

    def to(self, device: torch) -> None:
        self._device = device
        self._actor_network.to(self._device)

        if self._normalizer:
            self._normalizer.to(self._device)

    @staticmethod
    def load(path: str, actor_network: nn.Module) -> Policy:
        state_dict = torch.load(path, map_location="cpu")

        actor_network.load_state_dict(state_dict["actor"])
        normalizer = state_dict["normalizer"]

        return ActorBasedPolicy(actor_network, normalizer)


class ActorTanhGaussianBasedPolicy(Policy):
    def __init__(self, actor_network: TanhGaussianMLPNet) -> None:
        super().__init__()
        self._actor_network = actor_network

    def predict_actions(self, obs: List[Any]) -> List[Any]:
        obs     = _convert_to_torch(array=obs, device=self._device)
        actions = self.predict_actions_torch(obs)

        return actions.cpu().numpy()

    def predict_actions_torch(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Normalize if needed
            if self._normalizer:
                obs = self._normalizer.normalize_torch(obs)
            distr, _, _  = self._actor_network.forward(obs)

            return distr.mode

    def save(self, path: str) -> None:
        torch.save(self._actor_network.state_dict(), path)

    def to(self, device: torch) -> None:
        self._device = device
        self._actor_network.to(self._device)

        if self._normalizer:
            self._normalizer.to(self._device)

    @staticmethod
    def load(path: str, actor_network: nn.Module) -> Policy:
        actor_network.load_state_dict(torch.load(path, map_location="cpu"))
        
        return ActorTanhGaussianBasedPolicy(actor_network)


class ActorGMMBasedPolicy(Policy):
    def __init__(
        self, 
        actor_network: GMMMlpNet,
        normalizer: Optional[Normalizer]
    ) -> None:
        super().__init__()
        self._actor_network = actor_network
        self._normalizer    = normalizer

    def predict_actions(self, obs: List[Any]) -> List[Any]:
        obs     = _convert_to_torch(array=obs, device=self._device)
        actions = self.predict_actions_torch(obs)

        return actions.cpu().numpy()

    def predict_actions_torch(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Normalize if needed
            if self._normalizer:
                obs = self._normalizer.normalize_torch(obs)
                
            distr = self._actor_network(obs, low_std=True)

            if self._actor_network._num_modes == 1:
                return distr.mean
            else:
                return distr.sample()

    def save(self, path: str) -> None:
        torch.save({
            "actor": self._actor_network.state_dict(),
            "normalizer": self._normalizer,
            "num_modes": self._actor_network._num_modes
        }, path)

    def to(self, device: torch) -> None:
        self._device = device
        self._actor_network.to(self._device)

        if self._normalizer:
            self._normalizer.to(self._device)

    @staticmethod
    def load(path: str, actor_network: GMMMlpNet) -> Policy:
        state_dict = torch.load(path, map_location="cpu")

        # A bit of hacking the aRcHiTecTurE
        actor_network = GMMMlpNet(
            obs_dim=actor_network._obs_dim,
            action_dim=actor_network._action_dim,
            num_modes=state_dict["num_modes"]
        )

        actor_network.load_state_dict(state_dict["actor"])
        normalizer               = state_dict["normalizer"]

        return ActorGMMBasedPolicy(actor_network, normalizer)