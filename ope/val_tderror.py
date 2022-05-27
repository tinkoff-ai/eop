import warnings
from algs.core import Trainer
from policies.policy import Policy
import torch

from torch.utils.data.dataloader import DataLoader
from datasets.core import MDPDataset
from datasets.torch import FiniteMDPTorchDataset
from ope.core import OfflineEvalTrainer
from typing import Dict, Any, Optional


class ValTDError(OfflineEvalTrainer):

    def __init__(
        self,
        policy: Policy,
        policy_trainer: Trainer,
        hyperparams: Dict[str, Any]
    ):
        self._policy         = policy
        self._q_function     = policy_trainer.get_q_function()
        self._hyperparams    = hyperparams

        if self._q_function is None:
            warnings.warn("This OPE supports q-learning based algos only.")

    def to(self, device: torch.device) -> None:
        self._device = device
        self._policy.to(device)

    def eval(
        self,
        val_dataset: MDPDataset,
    ) -> Optional[float]:
        # Works only for q-learning based algos
        if self._q_function is None:
            return None

        loader = DataLoader(
            dataset    = FiniteMDPTorchDataset(val_dataset),
            batch_size = int(self._hyperparams["batch_size"])
        )

        # self._q_function.eval()

        num_samples = len(loader.dataset)
        with torch.no_grad():
            total = 0.0
            for batch in loader:
                act, obs         = batch["action"].to(self._device), batch["observation"].to(self._device)
                rew, next_obs    = batch["reward"].to(self._device), batch["next_observation"].to(self._device)
                new_next_act     = self._policy.predict_actions_torch(next_obs)

                # Simple bellman error
                total += torch.sum((self._q_function(obs, act) - (rew + self._q_function(next_obs, new_next_act))) ** 2)

        return (total / num_samples).cpu().item()


    def save(self, name: str) -> None:
        pass

    def load(self, path: str, name: str, device: torch.device = "cpu") -> None:
        pass