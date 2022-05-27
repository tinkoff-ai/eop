import os
import gym
import gym_minigrid
import numpy as np
import torch.nn as nn

from copy import deepcopy
from gym_minigrid.wrappers import FullyObsWrapper
from gym import ObservationWrapper
from policies.policy    import Policy, RandomPolicy
from policies.fourrooms import FourRoomsExpertPolicy
from typing import Optional, Tuple
from nets import ValueConvNet


class ImageObservationOnly(ObservationWrapper):
    def __init__(self, env):
        super(ImageObservationOnly, self).__init__(env)
        self.observation_space       = deepcopy(env.observation_space["image"])
        self.observation_space.shape = (self.observation_space.shape[-1], 
            self.observation_space.shape[0], self.observation_space.shape[1])

    def observation(self, observation):
        return observation["image"].transpose(2, 0, 1)


def create_env():
    env = ImageObservationOnly(
            FullyObsWrapper(
                gym.make("MiniGrid-FourRooms-v1", 
                    goal_pos=np.array([16, 2])
                )
            )
        )

    return env


# Expert-trajectories are heavy, so we cache them **shrug**
_cached_baseline: Optional[FourRoomsExpertPolicy] = None

def get_baseline(level: str) -> Policy:
    global _cached_baseline

    if level == "random":
        return RandomPolicy(create_env())
    else:
        expert_eps                  = float(level)
        cached_expert_trajectories  = os.path.join(os.environ["BASELINES_PATH"], "four-rooms-expert-solutions.pkl")

        # We cache the baseline so that solutions are loaded multiple times
        if _cached_baseline is None:
            _cached_baseline = FourRoomsExpertPolicy(
                expert_eps=expert_eps, 
                env=create_env(), 
                solutions_path=cached_expert_trajectories
            )
        else:
            _cached_baseline.set_expert_eps(expert_eps)
        
        return _cached_baseline


def get_networks(
    two_head: bool = False
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    #  Not for this env
    if two_head:
        raise NotImplementedError()

    env = create_env()
    obs_shape: np.ndarray = env.observation_space.shape
    n_actions: int        = env.action_space.n

    return ValueConvNet(obs_shape=obs_shape, n_actions=n_actions), None
    