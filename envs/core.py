import gym
import envs.fourrooms as fourrooms
import envs.finrl     as finrl
import envs.citylearn as citylearn
import envs.industrial as industrial
import torch.nn as nn

from policies.policy import Policy
from typing import Tuple, Optional


FOUR_ROOMS           = "four-rooms"
FIN_RL               = "finrl"
CITY_LEARN           = "citylearn"
INDUSTRIAL_BENCHMARK = "industrial"


def create_env(name: str) -> gym.Env:
    if name == FOUR_ROOMS:
        return fourrooms.create_env()
    elif name == FIN_RL:
        return finrl.create_env()
    elif name == CITY_LEARN:
        return citylearn.create_env()
    elif name == INDUSTRIAL_BENCHMARK:
        return industrial.create_env()
        
    raise NotImplementedError()


def get_baseline(env_name: str, level: str = "medium") -> Policy:
    if env_name == FOUR_ROOMS:
        return fourrooms.get_baseline(level=level)
    elif env_name == FIN_RL:
        return finrl.get_baseline(level=level)
    elif env_name == CITY_LEARN:
        return citylearn.get_baseline(level=level)
    elif env_name == INDUSTRIAL_BENCHMARK:
        return industrial.get_baseline(level=level)

    raise NotImplementedError()


def get_networks(
    env_name: str,
    two_head: bool = False,
    num_modes: Optional[int] = None
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    if env_name == FOUR_ROOMS:
        return fourrooms.get_networks(two_head)
    elif env_name == FIN_RL:
        return finrl.get_networks(
            two_head=two_head,
            num_modes=num_modes
        )
    elif env_name == CITY_LEARN:
        return citylearn.get_networks(
            two_head=two_head,
            num_modes=num_modes
        )
    elif env_name == INDUSTRIAL_BENCHMARK:
        return industrial.get_networks(
            two_head=two_head,
            num_modes=num_modes
        )

    raise NotImplementedError()