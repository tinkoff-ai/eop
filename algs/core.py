
from envs.core import FOUR_ROOMS
from typing import Any, Dict, Callable

import gym
import torch
from policies.policy import Policy
from wandb.wandb_run import Run
from utils.normalize import Normalizer
from typing import Optional
from pathlib import Path


ALG_NAME_CQL    = "cql"
ALG_NAME_TD3BC  = "td3+bc"
ALG_NAME_BC     = "bc"
ALG_NAME_FISHER = "fisherbrc"
ALG_NAME_CRR    = "crr"


class Trainer:
    _logger: Run
    _obs_normalizer: Optional[Normalizer] = None

    def set_logger(self, logger: Run) -> None:
        self._logger = logger

    def to(self, device: torch.device) -> None:
        raise NotImplementedError()

    def step(
        self, 
        batch: Dict[str, torch.Tensor], 
        artificial_batch: Any = None
    ) -> None:
        raise NotImplementedError()

    def on_epoch_end(self) -> None:
        raise NotImplementedError()

    def save_policy(self, name: str) -> None:
        raise NotImplementedError()

    def save(self, name: str) -> None:
        raise NotImplementedError()

    def load(self, path: str, device: torch.device = "cpu") -> None:
        raise NotImplementedError()

    def get_q_function(self) -> Optional[torch.nn.Module]:
        return None

    def get_policy_type(self):
        raise NotImplementedError()

    def set_normalizer(self, normalizer: Normalizer) -> None:
        self._obs_normalizer = normalizer


import algs.cql as cql
import algs.td3 as td3
import algs.bc  as bc


def create_trainer(
    alg_name: str,
    env_name: str,
    hyperparams: Dict[str, Any]
) -> Trainer:
    if alg_name == ALG_NAME_CQL:
        return cql.create_trainer(env_name, hyperparams)
    elif alg_name == ALG_NAME_TD3BC:
        return td3.create_trainer(env_name, hyperparams)
    elif alg_name == ALG_NAME_BC:
        return bc.create_trainer(env_name, hyperparams)
    else:
        raise NotImplementedError()


def load_trainer(
    alg_name   : str,
    env_name   : str,
    hyperparams: Dict[str, Any],
    path       : Path,
    device     : torch.device = "cpu"
) -> Trainer:
    trainer = create_trainer(alg_name, env_name, hyperparams)
    trainer.load(path, device)
    return trainer


def load_alg_policy(
    alg_name   : str,
    env_name   : str,
    policy_path: str
) -> Policy:
    if alg_name == ALG_NAME_CQL:
        return cql.load_alg_policy(env_name, policy_path)
    elif alg_name == ALG_NAME_TD3BC:
        return td3.load_alg_policy(env_name, policy_path)
    elif alg_name == ALG_NAME_BC:
        return bc.load_alg_policy(env_name, policy_path)
    else:
        raise NotImplementedError()


def sample_hyperparams(
    env_name: str,
    alg_name: str
) -> Dict[str, Any]:
    if alg_name == ALG_NAME_CQL:
        return cql.sample_hyperparams(env_name)
    elif alg_name == ALG_NAME_TD3BC:
        return td3.sample_hyperparams(env_name)
    elif alg_name == ALG_NAME_BC:
        return bc.sample_hyperparams(env_name)
    else:
        raise NotImplementedError()
