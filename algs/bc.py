"""
https://github.com/ARISE-Initiative/robomimic/blob/2804dd97dd1625ec861298a35cb677129d3bfacc/robomimic/models/policy_nets.py#L429
https://github.com/ARISE-Initiative/robomimic/blob/2804dd97dd1625ec861298a35cb677129d3bfacc/robomimic/models/obs_nets.py#L520
https://github.com/polixir/NeoRL/blob/benchmark/benchmark/OfflineRL/offlinerl/algo/modelfree/bc.py
https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/algo/bc.py
"""

from nets import GMMMlpNet
import os
from typing import Any, Dict

import gym
import gym.spaces
import algs.core as algs

import numpy as np
import torch
import torch.optim as optim

from envs.core import create_env, get_networks
from policies.policy import ActorGMMBasedPolicy
from policies.policy import Policy


class BC(algs.Trainer):
    def __init__(
        self,
        actor_network: GMMMlpNet,
        hyperparams: Dict[str, Any]
    ) -> None:
        super().__init__()

        self._hyperparams = hyperparams
        self._batch_count = 0
        self._epoch_count = 0

        self._actor           = actor_network
        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=hyperparams["actor_lr"]
        )

    def step(
        self,
        batch: Dict[str, torch.Tensor],
        artificial_batch: Any = None
    ) -> None:
        # Normalize first if needed
        if self._obs_normalizer:
            with torch.no_grad():
                batch["observation"] = self._obs_normalizer.normalize_torch(batch["observation"])
                batch["next_observation"] = self._obs_normalizer.normalize_torch(batch["next_observation"])

        # Do BC step
        action_dist = self._actor(batch["observation"])
        loss = -action_dist.log_prob(batch["action"]).mean()

        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        
        # Collect logs
        logs = {
            "loss/actor"        : loss,
            "counts/num_samples": (self._batch_count + 1) * self._hyperparams["batch_size"],
            "counts/num_epochs" : self._epoch_count,
            "counts/num_batches": self._batch_count+1
        }

        # Logging
        self._logger.log(logs)

        self._batch_count += 1

    def on_epoch_end(self) -> None:
        self._epoch_count += 1

    def to(self, device: torch.device) -> None:
        self._device = device
        self._actor.to(device)
        
        if self._obs_normalizer:
            self._obs_normalizer.to(device)

    def save_policy(self, name: str) -> None:
        policy      = ActorGMMBasedPolicy(self._actor, self._obs_normalizer)
        policy_path = os.path.join(self._logger.dir, f"{name}.policy")
        policy.save(path=policy_path)

        # Save to wandb as well
        self._logger.save(policy_path, policy="now")

    def save_model(self, name: str) -> None:
        base_path = self._logger.dir

        # save actor
        torch.save({ 
            "actor": self._actor.state_dict(),
            "actor_optim": self._actor_optimizer.state_dict(),
            "hyperparams": self._hyperparams,
            "obs_normalizer": self._obs_normalizer
        }, f=os.path.join(base_path, f"{name}.pt"))

    def load(self, path: str, device: torch.device = "cpu") -> None:
        state_dict   = torch.load(path, map_location=device)

        self._actor.load_state_dict(state_dict["actor"])
        self._actor_optimizer.load_state_dict(state_dict["actor_optim"])

        self._normalizer = state_dict["obs_normalizer"]
        self._hyperparams = state_dict["hyperparams"]

    def save(self, name: str) -> None:
        self.save_policy(name)
        self.save_model(name)


def create_trainer(
    env_name: str,
    hyperparams: Dict[str, Any]
) -> algs.Trainer:
    env = create_env(name=env_name)
    _, policy_net = get_networks(
        env_name, 
        num_modes=hyperparams["gmm_num_modes"]
    )

    if isinstance(env.action_space, gym.spaces.Box):
        return BC(actor_network=policy_net, hyperparams=hyperparams)
    else:
        raise NotImplementedError()


def load_alg_policy(
    env_name: str,
    policy_path: str
) -> Policy:
    env = create_env(name=env_name)
    # num_modes to invoke GMM
    _, policy_net = get_networks(env_name, num_modes=1)

    if isinstance(env.action_space, gym.spaces.Box):
        return ActorGMMBasedPolicy.load(path=policy_path, actor_network=policy_net)
    else:
        raise NotImplementedError()


def sample_hyperparams(
    env_name: str
) -> Dict[str, Any]:
    env = create_env(name=env_name)

    if isinstance(env.action_space, gym.spaces.Box):
        return {
            # Ours
            "batch_size"      : np.random.choice([256, 512]),
            # From What Matters in ...
            "gmm_num_modes": np.random.choice([1, 5, 10, 100]),
            # From What Matters in ...
            "n_gradient_steps": 2e5,
            # From TD3+BC paper
            "normalize_obs": True,
            # From What Matters in + NeoRL
            "actor_lr": np.random.choice([1e-3, 3e-4, 1e-4]),
        }
    else:
        raise NotImplementedError()
