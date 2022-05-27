from policies.policy import Policy
import torch

from torch.utils.data.dataloader import DataLoader
from datasets.core import MDPDataset
from datasets.torch import FiniteMDPTorchDataset
from ope.core import OfflineEvalTrainer
from typing import Dict, Any


class ValContinuousActionDiff(OfflineEvalTrainer):

    def __init__(
        self,
        policy: Policy,
        hyperparams: Dict[str, Any]
    ):
        self._policy      = policy
        self._hyperparams = hyperparams

    def to(self, device: torch.device) -> None:
        self._device = device
        self._policy.to(device)

    def eval(
        self,
        val_dataset: MDPDataset,
    ) -> float:
        loader = DataLoader(
            dataset    = FiniteMDPTorchDataset(val_dataset),
            batch_size = int(self._hyperparams["batch_size"])
        )

        total_diff = 0.0
        num_samples = len(loader.dataset)
        for batch in loader:
            act, obs    = batch["action"].to(self._device), batch["observation"].to(self._device)
            new_act     = self._policy.predict_actions_torch(obs)
            total_diff += torch.sum((act - new_act)**2)

        return (total_diff / num_samples).cpu().item()


    def save(self, name: str) -> None:
        pass

    def load(self, path: str, name: str, device: torch.device = "cpu") -> None:
        pass