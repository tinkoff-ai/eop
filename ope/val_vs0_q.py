import warnings
from algs.core import Trainer
from policies.policy import Policy
import torch

from torch.utils.data.dataloader import DataLoader
from datasets.core import MDPDataset
from datasets.torch import FiniteMDPTorchDataset
from ope.core import OfflineEvalTrainer
from typing import Dict, Any, Optional


class ValVs0TrainedQ(OfflineEvalTrainer):

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
            dataset    = FiniteMDPTorchDataset(val_dataset, s0_only=True),
            batch_size = int(self._hyperparams["batch_size"]),
        )

        num_samples = len(loader.dataset)
        with torch.no_grad():
            total = 0.0
            for batch in loader:
                obs         = batch["observation"].to(self._device)
                policy_act  = self._policy.predict_actions_torch(obs)
                total      += torch.sum(self._q_function(obs, policy_act))


        return (total / num_samples).cpu().item()


    def save(self, name: str) -> None:
        pass

    def load(self, path: str, name: str, device: torch.device = "cpu") -> None:
        pass