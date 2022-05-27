
from algs.core import Trainer
from envs.core import FOUR_ROOMS
from typing import Any, Dict, Callable

import gym
import torch
from policies.policy import Policy
from wandb.wandb_run import Run
from utils.normalize import Normalizer
from typing import Optional
from pathlib import Path
from datasets.core import MDPDataset


OPE_NAME_VAL_LOSS     = "val_loss"
OPE_NAME_VAL_TD_ERROR = "val_tderror"
OPE_NAME_VAL_S0_Q     = "val_vs0_q"
OPE_NAME_VAL_S0_FQE   = "val_vs0_fqe"


class OfflineEvalTrainer:
    _obs_normalizer: Optional[Normalizer] = None

    def set_logger(self, logger: Run) -> None:
        self._logger = logger

    def to(self, device: torch.device) -> None:
        raise NotImplementedError()

    def train(
        self,
        train_dataset: Optional[MDPDataset],
        val_dataset  : MDPDataset,
    ) -> None:
        pass

    def eval(
        self,
        val_dataset: MDPDataset,
    ) -> Optional[float]:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        raise NotImplementedError()

    def load(self, path: str, device: torch.device = "cpu") -> None:
        raise NotImplementedError()

    def set_normalizer(self, normalizer: Normalizer) -> None:
        self._obs_normalizer = normalizer


from ope.val_action_diff import ValContinuousActionDiff
from ope.val_tderror import ValTDError
from ope.val_vs0_q import ValVs0TrainedQ
from ope.val_vs0_fqe import ValVs0FQE


def create_eval_trainer(
    alg_name: str,
    policy_trainer: Trainer,
    policy: Policy,
    train_config: Dict[str, Any]
) -> OfflineEvalTrainer:
    if alg_name == OPE_NAME_VAL_LOSS:
        return ValContinuousActionDiff(
            policy      = policy,
            hyperparams = train_config
        )
    elif alg_name == OPE_NAME_VAL_TD_ERROR:
        return ValTDError(
            policy         = policy,
            policy_trainer = policy_trainer,
            hyperparams    = train_config
        )
    elif alg_name == OPE_NAME_VAL_S0_Q:
        return ValVs0TrainedQ(
            policy         = policy,
            policy_trainer = policy_trainer,
            hyperparams    = train_config
        )
    elif alg_name == OPE_NAME_VAL_S0_FQE:
        return ValVs0FQE(
            policy      = policy,
            env_name    = train_config["env"],
            hyperparams = train_config
        )
    else:
        raise NotImplementedError()


def sample_hyperparams(
    alg_name: str
) -> Dict[str, Any]:
    if alg_name != OPE_NAME_VAL_S0_FQE:
        return {
            "batch_size": 512
        }
    elif alg_name == OPE_NAME_VAL_S0_FQE:
        return {
            # From Paine et al. 2020 and NeoRL
            "batch_size"            : 256,
            "n_gradient_steps"      : 250000,
            "target_update_interval": 100,
            "gamma"                 : 0.99,
            "critic_lr"             : 1e-4,
            "polyak"                : 0.0
        }