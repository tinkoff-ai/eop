import neorl
import numpy as np
import os
import torch.nn as nn

from typing import Tuple, Optional
from policies.policy import Policy, RandomPolicy
from policies.stable_baselines3 import SB3BasedPolicy
from stable_baselines3 import SAC
from nets import ValueMLPNet, TanhGaussianMLPNet
from nets import TwoHeadValueMLPNet, TanhClampedMLPNet, GMMMlpNet


def create_env():
    return neorl.make("citylearn")


def get_baseline(level: str) -> Policy:
    if level == "random":
        return RandomPolicy(create_env())
    else:
        policy_path = os.path.join(os.environ["BASELINES_PATH"], "citylearn", f"{level}.zip")
        if not os.path.exists(policy_path):
            raise Exception("No such policy.")

        return SB3BasedPolicy(
            sb3_model=SAC.load(policy_path)
        )


def get_networks(
    two_head : bool = False,
    num_modes: Optional[int] = None,
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    env              = create_env()
    obs_dim  : int   = env.observation_space.shape[0]
    n_actions: int   = env.action_space.shape[0]
    act_range: float = env.action_space.high[0]

    # TD3+BC
    if two_head:
        actor_net = TanhClampedMLPNet(obs_dim, n_actions, act_range)
        value_net = TwoHeadValueMLPNet(obs_dim, n_actions)
    else:
        # CQL
        if not num_modes:
            value_net = ValueMLPNet(state_dim=obs_dim, action_dim=n_actions)
            actor_net = TanhGaussianMLPNet(obs_dim, n_actions)
        # Behavioral Clonning
        else:
            value_net = None
            actor_net = GMMMlpNet(obs_dim, n_actions, num_modes)

    return value_net, actor_net