"""
https://github.com/polixir/NeoRL/blob/benchmark/benchmark/OfflineRL/offlinerl/algo/modelfree/cql.py
https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py
"""

import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import gym
import gym.spaces
import math
import algs.core as algs

from numpy.lib.arraysetops import isin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.torch import TorchMiniBatch
from envs.core import create_env, get_networks
from policies.policy import ValueBasedPolicy, ActorTanhGaussianBasedPolicy
from torch.nn.modules import loss
from utils.torch import hard_sync, compute_huber_loss, soft_sync
from policies.policy import Policy


CQL_VARIANT_ENTROPY = "entropy"
CQL_VARIANT_KL      = "kl"


class DiscreteCQL(algs.Trainer):
    def __init__(
        self,
        value_network: nn.Module,
        hyperparams: Dict[str, Any]
    ) -> None:
        super().__init__()

        self._hyperparams = hyperparams
        self._batch_count = 0
        self._epoch_count = 0

        # Create networks
        self._value_network        = value_network
        self._target_value_network = deepcopy(value_network)

        # Create optimizer
        self._optim = optim.Adam(
            params=self._value_network.parameters(),
            lr=hyperparams["lr"]
        )
        
    def step(
        self, 
        batch: TorchMiniBatch, 
        artificial_batch: Any = None
    ) -> None:
        self._optim.zero_grad()

        # Compute bootstrapped targets (ddqn-style)
        with torch.no_grad():
            actions   = self._value_network(batch["next_observation"]).argmax(dim=1, keepdim=True)
            ns_values = self._target_value_network(batch["next_observation"]).gather(dim=1, index=actions)
            y_target  = batch["reward"] + (self._hyperparams["gamma"] * ns_values * (1.0 - batch["terminal"]))

        # Compute MSE on bootstrapped targets
        policy_values = self._value_network(batch["observation"])
        y_predicted   = policy_values.gather(dim=1, index=batch["action"].long().unsqueeze(-1))
        loss_bellman  = torch.mean(compute_huber_loss(y=y_predicted, target=y_target))

        # Compute conservative part
        loss_conservative  = (torch.logsumexp(policy_values, dim=1, keepdim=True) - y_predicted).mean()

        # Optimize
        loss_total = self._hyperparams["alpha"] * loss_conservative + loss_bellman
        loss_total.backward()
        self._optim.step()

        # Logging
        self._logger.log({
            "loss/total": loss_total,
            "loss/bellman": loss_bellman,
            "loss/conservative": loss_conservative,
            "counts/num_samples": (self._batch_count + 1) * self._hyperparams["batch_size"]
        })

        self._batch_count += 1

    def on_epoch_end(self) -> None:
        self._epoch_count += 1

        # Target network update
        if self._epoch_count % self._hyperparams["target_update_interval"] == 0:
            hard_sync(self._target_value_network, self._value_network)


    def to(self, device: torch.device) -> None:
        self._value_network.to(device)
        self._target_value_network.to(device)

    def save_policy(self, name: str) -> None:
        # Create a policy based on the learned value network and save it
        policy = ValueBasedPolicy(self._value_network)
        policy_path = os.path.join(self._logger.dir, name)
        policy.save(path=policy_path)

        # Save to wandb as well
        self._logger.save(policy_path, policy="now")


class ContinuousCQL(algs.Trainer):
    """
    Monstrous.
    """
    def __init__(
        self,
        env            : gym.Env,
        critic_networks: Tuple[nn.Module, nn.Module],
        actor_network  : nn.Module,
        hyperparams    : Dict[str, Any]
    ) -> None:
        super().__init__()

        self._hyperparams = hyperparams
        self._batch_count = 0
        self._epoch_count = 0

        # SAC temperature adjustment
        self._sac_log_alpha       = torch.zeros(1, requires_grad=True)
        self._sac_log_alpha_optim = optim.Adam(
            params=[self._sac_log_alpha],
            lr=hyperparams["sac_lr"]
        )
        self._sac_target_entropy: float = -np.prod(env.action_space.shape).item()

        # Actor
        self._actor       = actor_network
        self._actor_optim = optim.Adam(
            params=self._actor.parameters(),
            lr=hyperparams["sac_lr"]
        )

        # Critics
        self._critic1        = critic_networks[0]
        self._critic1_target = deepcopy(self._critic1)
        self._critic1_opt    = optim.Adam(
            params=self._critic1.parameters(),
            lr=hyperparams["cql_lr"]
        )

        self._critic2        = critic_networks[1]
        self._critic2_target = deepcopy(self._critic2)
        self._critic2_opt    = optim.Adam(
            params=self._critic2.parameters(),
            lr=hyperparams["cql_lr"]
        )

        # CQL alpha adjustment
        self._cql_log_alpha = torch.zeros(1, requires_grad=True)
        self._cql_log_alpha_optim = optim.Adam(
            params=[self._cql_log_alpha],
            lr=hyperparams["cql_lr"]
        )

    def _sample_actions(
        self, 
        observations: torch.Tensor, 
        log_prob: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, actions, log_probs = self._actor(observations)
        return actions, log_probs

    def _sample_n_actions(
        self, 
        observations: torch.Tensor,
        num_actions : int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stacked_observations = observations.unsqueeze(1).repeat(1, num_actions, 1).view(observations.shape[0] * num_actions, observations.shape[1])
        stacked_actions, stacked_log_probs = self._sample_actions(
            observations = stacked_observations,
            log_prob     = True
        )

        return stacked_actions, stacked_log_probs.view(observations.shape[0], num_actions, 1)

    def _get_critic_values(
        self,
        critic: nn.Module,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        # Infer number of repeated actions (it should be cql_n_uniform_actions)
        action_shape = actions.shape[0]
        obs_shape    = observations.shape[0]
        num_repeat   = int(action_shape / obs_shape)

        # Repeat observations to match shape with actions
        obs_temp = observations.unsqueeze(1).repeat(1, num_repeat, 1).view(observations.shape[0] * num_repeat, observations.shape[1])
        
        # Compute q-values
        q_values = critic(obs_temp, actions).view(observations.shape[0], num_repeat, 1)

        return q_values
  
    def step(
        self, 
        batch: TorchMiniBatch, 
        artificial_batch: Any = None
    ) -> None:
        # Compute actions and their logprobs
        actions, log_probs = self._sample_actions(batch["observation"], log_prob=True)

        ### SAC: Tune alpha
        sac_alpha      = 1.0
        sac_alpha_loss = None
        if self._hyperparams["sac_alpha_tuning"]:
            # Update SAC alpha
            self._sac_log_alpha_optim.zero_grad()
            sac_alpha_loss = -(self._sac_log_alpha * (log_probs + self._sac_target_entropy).detach()).mean()
            sac_alpha_loss.backward()
            self._sac_log_alpha_optim.step()

            # Use updated alpha
            sac_alpha = self._sac_log_alpha.exp()

        ### SAC: Actor update
        self._actor_optim.zero_grad()
        actor_loss = (
            -torch.min(
                self._critic1(batch["observation"], actions), 
                self._critic2(batch["observation"], actions)
            ) + sac_alpha * log_probs).mean()
        actor_loss.backward()
        self._actor_optim.step()

        ### CQL: Q-Function Bellman Error compute
        q1_actions = self._critic1(batch["observation"], batch["action"])
        q2_actions = self._critic2(batch["observation"], batch["action"])

        next_actions, next_log_probs = self._sample_actions(batch["next_observation"], log_prob=True)
        if self._hyperparams["cql_max_backup"]:
            next_actions_temp, _ = self._sample_n_actions(
                batch["next_observation"], 
                num_actions=10, 
            )
            q1_next_actions = self._get_critic_values(
                critic=self._critic1,
                observations=batch["next_observation"], 
                actions=next_actions_temp
            ).max(1)[0].view(-1, 1)
            q2_next_actions = self._get_critic_values(
                critic=self._critic2,
                observations=batch["next_observation"], 
                actions=next_actions_temp
            ).max(1)[0].view(-1, 1)
            q_next_actions = torch.min(q1_next_actions, q2_next_actions)
        else:
            q_next_actions = torch.min(
                self._critic1_target(batch["next_observation"], next_actions),
                self._critic2_target(batch["next_observation"], next_actions)
            )
            q_next_actions = q_next_actions - sac_alpha * next_log_probs

        q_actions_target          = batch["reward"].unsqueeze(-1) + (1. - batch["terminal"].unsqueeze(-1) ) * self._hyperparams["gamma"] * q_next_actions.detach()
        critic1_loss_bellman_part = ((q1_actions - q_actions_target)**2).mean()
        critic2_loss_bellman_part = ((q2_actions - q_actions_target)**2).mean()

        ### CQL: Q-Function Conservative compute
        # Importance Sampling: uniformly sample actions
        is_uniform_actions = torch.FloatTensor(
            actions.shape[0] * self._hyperparams["cql_n_uniform_actions"], 
            actions.shape[-1]
        ).uniform_(-1, 1).to(self._device)
        # Importance Sampling: sample actions from current policy
        is_actions, is_log_probs = self._sample_n_actions(
            observations=batch["observation"],
            num_actions=self._hyperparams["cql_n_uniform_actions"]
        )
        # Importance Sampling: sample actions from current policy in next states
        # it was in the codebase but not in the paper *shrug*
        is_next_actions, is_next_log_probs = self._sample_n_actions(
            observations=batch["next_observation"],
            num_actions=self._hyperparams["cql_n_uniform_actions"]
        )

        # Importance Sampling: Compute q-values for sampled actions
        q1_is_uniform_actions = self._get_critic_values(
            critic=self._critic1,
            observations=batch["observation"],
            actions=is_uniform_actions
        )
        q1_is_actions = self._get_critic_values(
            critic=self._critic1,
            observations=batch["observation"],
            actions=is_actions
        )
        q1_is_next_actions = self._get_critic_values(
            critic=self._critic1,
            observations=batch["next_observation"],
            actions=is_next_actions
        )

        q2_is_uniform_actions = self._get_critic_values(
            critic=self._critic2,
            observations=batch["observation"],
            actions=is_uniform_actions
        )
        q2_is_actions = self._get_critic_values(
            critic=self._critic2,
            observations=batch["observation"],
            actions=is_actions
        )
        q2_is_next_actions = self._get_critic_values(
            critic=self._critic2,
            observations=batch["next_observation"],
            actions=is_next_actions
        )

        # Importance Sampling: Compute conservative part of the loss
        if self._hyperparams["cql_variant"] == CQL_VARIANT_ENTROPY:
            q1_is_all = torch.cat([
                q1_is_uniform_actions - np.log(0.5 ** actions.shape[-1]),
                q1_is_actions - is_log_probs.detach(),
                q1_is_next_actions - is_next_log_probs.detach()
            ], dim=1)
            q2_is_all = torch.cat([
                q2_is_uniform_actions - np.log(0.5 ** actions.shape[-1]),
                q2_is_actions - is_log_probs.detach(),
                q2_is_next_actions - is_next_log_probs.detach()
            ], dim=1)
        elif self._hyperparams["cql_variant"] == CQL_VARIANT_KL:
            q1_is_all = torch.cat([
                q1_is_uniform_actions,
                q1_actions.unsqueeze(1),
                q1_is_actions,
                q1_is_next_actions,
            ], dim=1)
            q2_is_all = torch.cat([
                q2_is_uniform_actions,
                q2_actions.unsqueeze(1),
                q2_is_actions,
                q2_is_next_actions,
            ], dim=1)
        else:
            raise NotImplementedError()
        
        critic1_loss_cons_part = torch.logsumexp(
            q1_is_all / self._hyperparams["cql_temp"], 
            dim=1
        ).mean() * self._hyperparams["cql_min_q_weight"] * self._hyperparams["cql_temp"]
        critic1_loss_cons_part -= q1_actions.mean() * self._hyperparams["cql_min_q_weight"]

        critic2_loss_cons_part = torch.logsumexp(
            q2_is_all / self._hyperparams["cql_temp"], 
            dim=1
        ).mean() * self._hyperparams["cql_min_q_weight"] * self._hyperparams["cql_temp"]
        critic2_loss_cons_part -= q2_actions.mean() * self._hyperparams["cql_min_q_weight"]


        ### CQL: Tune alpha
        alpha_prime      = 1.0
        alpha_prime_loss = 0.0
        if self._hyperparams["cql_lagrange_thresh"] > 0.0:
            # Compute loss for the cql_alpha
            alpha_prime            = torch.clamp(self._cql_log_alpha.exp(), min=0.0, max=1000000.0)
            critic1_loss_cons_part = alpha_prime * (critic1_loss_cons_part - self._hyperparams["cql_lagrange_thresh"])
            critic2_loss_cons_part = alpha_prime * (critic2_loss_cons_part - self._hyperparams["cql_lagrange_thresh"])

            # Update alpha
            self._cql_log_alpha_optim.zero_grad()
            alpha_prime_loss = (-critic1_loss_cons_part - critic2_loss_cons_part)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self._cql_log_alpha_optim.step()

        ### FINAL LOSS
        critic1_loss = critic1_loss_bellman_part + critic1_loss_cons_part
        critic2_loss = critic2_loss_bellman_part + critic2_loss_cons_part

        ### Update critics....
        self._critic1_opt.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self._critic1_opt.step()

        self._critic2_opt.zero_grad()
        critic2_loss.backward()
        self._critic2_opt.step()

        ### Soft update target networks
        soft_sync(self._critic1_target, self._critic1, self._hyperparams["cql_tau"])
        soft_sync(self._critic2_target, self._critic2, self._hyperparams["cql_tau"])

        # Log...
        self._logger.log({
            "cql/loss/critic1/total": critic1_loss,
            "cql/loss/critic1/conservative": critic1_loss_cons_part,
            "cql/loss/critic1/bellman": critic1_loss_bellman_part,
            
            "cql/loss/critic2/total": critic2_loss,
            "cql/loss/critic2/conservative": critic2_loss_cons_part,
            "cql/loss/critic2/bellman": critic2_loss_bellman_part,

            "cql/loss/alpha": alpha_prime_loss,
            "cql/alpha": alpha_prime,

            "sac/loss/actor": actor_loss,
            "sac/loss/alpha": sac_alpha_loss,
            "sac/alpha": sac_alpha,

            "counts/num_samples": (self._batch_count + 1) * self._hyperparams["batch_size"],
            "counts/num_epochs": self._epoch_count
        })

        self._batch_count += 1

    def on_epoch_end(self) -> None:
        self._epoch_count += 1

    def get_q_function(self) -> Optional[torch.nn.Module]:
        return self._critic1

    def to(self, device: torch.device) -> None:
        self._device = device
        self._actor.to(device)
        self._critic1.to(device)
        self._critic1_target.to(device)
        self._critic2.to(device)
        self._critic2_target.to(device)

        # tensors must be re-assigned.....
        self._cql_log_alpha = self._cql_log_alpha.to(device)
        self._sac_log_alpha = self._sac_log_alpha.to(device)

    def save_policy(self, name: str) -> None:
        policy = ActorTanhGaussianBasedPolicy(self._actor)
        policy_path = os.path.join(self._logger.dir, f"{name}.policy")
        policy.save(path=policy_path)

        # Save to wandb as well
        self._logger.save(policy_path, policy="now")

    def save_model(self, name: str) -> None:
        base_path = self._logger.dir

        # save actor
        torch.save({ 
            "actor": self._actor.state_dict(),
            "critic1": self._critic1.state_dict(),
            "critic2": self._critic2.state_dict(),
            "critic1_target": self._critic1_target.state_dict(),
            "critic2_target": self._critic2_target.state_dict(),
            "critic1_opt": self._critic1_opt.state_dict(),
            "critic2_opt": self._critic2_opt.state_dict(),
            "sac_log_alpha": self._sac_log_alpha,
            "sac_log_alpha_optim": self._sac_log_alpha_optim.state_dict(),
            "cql_log_alpha": self._cql_log_alpha,
            "cql_log_alpha_optim": self._cql_log_alpha_optim.state_dict(),
            "hyperparams": self._hyperparams
        }, f=os.path.join(base_path, f"{name}.pt"))

    def load(self, path: str, device: torch.device = "cpu") -> None:
        state_dict   = torch.load(path, map_location=device)

        self._actor.load_state_dict(state_dict=state_dict["actor"])
        self._critic1.load_state_dict(state_dict=state_dict["critic1"])
        self._critic2.load_state_dict(state_dict=state_dict["critic2"])

        self._critic1_target.load_state_dict(state_dict=state_dict["critic1_target"])
        self._critic2_target.load_state_dict(state_dict=state_dict["critic2_target"])

        self._critic1_opt.load_state_dict(state_dict=state_dict["critic1_opt"])
        self._critic2_opt.load_state_dict(state_dict=state_dict["critic2_opt"])

        self._sac_log_alpha = state_dict["sac_log_alpha"]
        self._sac_log_alpha_optim.load_state_dict(state_dict=state_dict["sac_log_alpha_optim"])

        self._cql_log_alpha = state_dict["cql_log_alpha"]
        self._cql_log_alpha_optim.load_state_dict(state_dict=state_dict["cql_log_alpha_optim"])

        self._hyperparams = state_dict["hyperparams"]

    def save(self, name: str) -> None:
        self.save_policy(name)
        self.save_model(name)


def create_trainer(
    env_name: str,
    hyperparams: Dict[str, Any]
) -> algs.Trainer:
    env                  = create_env(name=env_name)
    val_net , policy_net = get_networks(env_name)
    val_net1, _          = get_networks(env_name)

    if isinstance(env.action_space, gym.spaces.Discrete):
        return DiscreteCQL(value_network=val_net, hyperparams=hyperparams)
    else:
        return ContinuousCQL(
            env             = env,
            critic_networks = (val_net, val_net1),
            actor_network   = policy_net,
            hyperparams     = hyperparams
        )

    
def load_alg_policy(
    env_name   : str,
    policy_path: str
) -> Policy:
    env                 = create_env(name=env_name)
    val_net, policy_net = get_networks(env_name)

    if isinstance(env.action_space, gym.spaces.Discrete):
        return ValueBasedPolicy.load(policy_path, val_net)
    else:
        return ActorTanhGaussianBasedPolicy.load(policy_path, policy_net)


def sample_hyperparams(
    env_name: str
) -> Dict[str, Any]:
    env = create_env(name=env_name)

    if isinstance(env.action_space, gym.spaces.Discrete):
        raise NotImplementedError()
        return {
            "batch_size"            : np.random.choice([32, 64, 128, 256, 512]),
            "gamma"                 : 0.9 + np.random.rand() * 0.1,
            "alpha"                 : np.random.choice([1, 5, 10]),
            "target_update_interval": np.random.choice([1, 2, 4, 10]),
            "lr"                    : np.random.choice([0.0000625, 0.0000625*10, 0.0000625*1e2]),
            "n_epochs"              : 40
        }
    else:
        return {
            # NeoRL
            "sac_alpha_tuning"     : np.random.choice([False, True]),
            # CQL original
            "sac_lr"               : np.random.choice([3e-5, 3e-4, 1e-4, 1e-3]),
            # NeoRL
            "cql_lagrange_thresh"  : np.random.choice([-1.0, 2.0, 5.0, 10.0]),
            # NeoRL
            "cql_temp"   : 1.0,
            "cql_variant": np.random.choice([CQL_VARIANT_ENTROPY, CQL_VARIANT_KL]),
            # NeoRL
            "cql_min_q_weight"     : np.random.choice([5, 10]),
            # CQL original
            "cql_lr"               : np.random.choice([1e-4, 3e-4, 1e-3]),
            # NeoRL?
            "cql_tau"              : np.random.choice([5e-3, 1e-2]),
            # NeoRL
            "cql_max_backup"       : np.random.choice([False, True]),
            # CQL original
            "cql_n_uniform_actions": 10,
            # CQL original + ours
            "batch_size"           : np.random.choice([256, 512]),
            # NeoRL
            "n_gradient_steps"     : 3e5,
            # Ours
            "gamma"                : 0.9 + np.random.rand() * 0.1,
        }
