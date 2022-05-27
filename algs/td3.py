"""
https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py
"""

import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import gym
import gym.spaces
import algs.core as algs

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets.torch import TorchMiniBatch
from envs.core import create_env, get_networks
from policies.policy import ValueBasedPolicy, ActorBasedPolicy
from utils.torch import soft_sync
from policies.policy import Policy


class TD3PlusBC(algs.Trainer):
    def __init__(
        self,
        actor_network: nn.Module,
        critic_network: nn.Module,
        max_action: float,
        hyperparams: Dict[str, Any]
    ) -> None:
        super().__init__()

        self._hyperparams = hyperparams
        self._batch_count = 0
        self._epoch_count = 0

        self._actor           = actor_network
        self._actor_target    = deepcopy(self._actor)
        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=hyperparams["actor_lr"]
        )

        self._critic           = critic_network
        self._critic_target    = deepcopy(self._critic)
        self._critic_optimizer = optim.Adam(
            self._critic.parameters(),
            lr=hyperparams["critic_lr"]
        )

        self._max_action = max_action

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

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(batch["action"]) * self._hyperparams["policy_noise"]
            ).clamp(-self._hyperparams["noise_clip"], self._hyperparams["noise_clip"])

            next_action = (
                self._actor_target(batch["next_observation"]) + noise
            ).clamp(-self._max_action, self._max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self._critic_target(
                batch["next_observation"], next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = batch["reward"].unsqueeze(-1) + (1. - batch["terminal"].unsqueeze(-1)) * \
                                        self._hyperparams["gamma"] * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self._critic(batch["observation"], batch["action"])

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                                 F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        # Collect logs
        logs = {
            "loss/critic"       : critic_loss,
            "counts/num_samples": (self._batch_count + 1) * self._hyperparams["batch_size"],
            "counts/num_epochs" : self._epoch_count,
            "counts/num_batches": self._batch_count+1
        }

        # Delayed policy updates
        if self._batch_count % self._hyperparams["policy_update_frequency"] == 0:

            # Compute actor loss
            pi    = self._actor(batch["observation"])
            Q     = self._critic.Q1(batch["observation"], pi)
            lmbda = self._hyperparams["alpha"]/Q.abs().mean().detach()

            bc_loss        = F.mse_loss(pi, batch["action"])
            td3_actor_loss = -lmbda * Q.mean()
            actor_loss     = td3_actor_loss + bc_loss

            # Optimize the actor
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            self._actor_optimizer.step()

            # Update the frozen target models
            soft_sync(self._critic_target, self._critic, self._hyperparams["tau"])
            soft_sync(self._actor_target, self._actor, self._hyperparams["tau"])

            # Add actor loss
            logs["loss/actor"]     = actor_loss
            logs["loss/actor_bc"]  = bc_loss
            logs["loss/actor_td3"] = td3_actor_loss

        # Logging
        self._logger.log(logs)

        self._batch_count += 1

    def on_epoch_end(self) -> None:
        self._epoch_count += 1

    def get_q_function(self) -> Optional[torch.nn.Module]:
        return self._critic.Q1

    def to(self, device: torch.device) -> None:
        self._device = device
        self._actor.to(device)
        self._actor_target.to(device)
        self._critic.to(device)
        self._critic_target.to(device)
        
        if self._obs_normalizer:
            self._obs_normalizer.to(device)

    def save_policy(self, name: str) -> None:
        policy      = ActorBasedPolicy(self._actor, self._obs_normalizer)
        policy_path = os.path.join(self._logger.dir, f"{name}.policy")
        policy.save(path=policy_path)

        # Save to wandb as well
        self._logger.save(policy_path, policy="now")

    def save_model(self, name: str) -> None:
        base_path = self._logger.dir

        # save actor
        torch.save({ 
            "actor": self._actor.state_dict(),
            "actor_target": self._actor_target.state_dict(),
            "actor_optim": self._actor_optimizer.state_dict(),
            "critic": self._critic.state_dict(),
            "critic_target": self._critic_target.state_dict(),
            "critic_optim": self._critic_optimizer.state_dict(),
            "hyperparams": self._hyperparams,
            "obs_normalizer": self._obs_normalizer
        }, f=os.path.join(base_path, f"{name}.pt"))

    def load(self, path: str, device: torch.device = "cpu") -> None:
        state_dict   = torch.load(path, map_location=device)

        self._actor.load_state_dict(state_dict["actor"])
        self._actor_target.load_state_dict(state_dict["actor_target"])
        self._actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self._critic.load_state_dict(state_dict["critic"])
        self._critic_target.load_state_dict(state_dict["critic_target"])
        self._critic_optimizer.load_state_dict(state_dict["critic_optim"])

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
    val_net, policy_net = get_networks(env_name, two_head=True)

    if isinstance(env.action_space, gym.spaces.Box):
        # This is totally okay for current set of environments (finrl, citylearn, ib)
        # As their action space is Box and equally bounded from both sides
        max_action = env.action_space.high[0]

        return TD3PlusBC(
            actor_network=policy_net,
            critic_network=val_net,
            max_action=max_action,
            hyperparams=hyperparams
        )
    else:
        raise NotImplementedError()


def load_alg_policy(
    env_name: str,
    policy_path: str
) -> Policy:
    env = create_env(name=env_name)
    _, policy_net = get_networks(env_name, two_head=True)

    if isinstance(env.action_space, gym.spaces.Box):
        return ActorBasedPolicy.load(path=policy_path, actor_network=policy_net)
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
            "gamma"           : 0.9 + np.random.rand() * 0.1,
            "n_gradient_steps": 3e5,

            # From TD3+BC paper
            "alpha"                  : np.random.choice([1.0, 2.0, 2.5, 3.0, 4.0]),
            "normalize_obs"          : True,
            "policy_noise"           : 0.2,
            "noise_clip"             : 0.5,
            "policy_update_frequency": 2,

            # From TD3+BC + match CQL
            "tau"      : np.random.choice([5e-3, 1e-2]),
            "actor_lr" : np.random.choice([3e-5, 3e-4, 1e-4, 1e-3]),
            "critic_lr": np.random.choice([1e-4, 3e-4, 1e-3]),
        }
    else:
        raise NotImplementedError()
