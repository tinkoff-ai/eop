from utils.torch import soft_sync

from policies.policy import Policy
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader
from utils.loader import BatchPrefetchLoaderWrapper
from datasets.core import MDPDataset
from datasets.torch import FiniteMDPTorchDataset, InfiniteMDPTorchDataset
from ope.core import OfflineEvalTrainer
from typing import Dict, Any, Optional
from copy import deepcopy
from envs.core import create_env


# As in NeoRL and Paine et al. 2020
class FQENetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ):
        super(FQENetwork, self).__init__()

        self._net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(
        self,
        obs   : torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        inp = torch.cat([obs, action], dim=1)
        return self._net(inp)


class ValVs0FQE(OfflineEvalTrainer):
    def __init__(
        self,
        policy: Policy,
        env_name: str,
        hyperparams: Dict[str, Any]
    ):
        # Extract problem dimensions...
        # Valid only for NeoRL envs...
        env = create_env(env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self._policy      = policy
        self._hyperparams = hyperparams
        self._critic      = FQENetwork(obs_dim + act_dim)
        self._critic_targ = deepcopy(self._critic)
        self._critic_targ.requires_grad_(False)
        self._loss_function   = nn.MSELoss()

        self._critic_optimizer = optim.Adam(
            params=self._critic.parameters(),
            lr=self._hyperparams["critic_lr"]
        )

    def to(self, device: torch.device) -> None:
        self._device = device
        self._policy.to(device)
        self._critic.to(device)
        self._critic_targ.to(device)

    def train(
        self,
        train_dataset: Optional[MDPDataset],
        val_dataset  : MDPDataset,
    ) -> None:
        val_iterator = BatchPrefetchLoaderWrapper(
            loader = DataLoader(
                dataset     = InfiniteMDPTorchDataset(val_dataset),
                batch_size  = int(self._hyperparams["batch_size"]),
                # num_workers = 16
            ),
            device=self._device,
            num_prefetches=100
        )

        # FQE clipping trick
        min_reward = min(val_dataset.rewards)
        max_reward = max(val_dataset.rewards)
        max_value  = (1.2 * max_reward + 0.8 * min_reward) / (1 - self._hyperparams["gamma"])
        min_value  = (1.2 * min_reward + 0.8 * max_reward) / (1 - self._hyperparams["gamma"])

        for ind, batch in enumerate(iter(val_iterator)):
            # Bellman error for the given policy
            with torch.no_grad():
                next_action = self._policy.predict_actions_torch(batch["next_observation"])
                y_true = batch["reward"].unsqueeze(-1)  + self._hyperparams["gamma"] * (1.0 - batch["terminal"].unsqueeze(-1) ) * self._critic_targ(batch["next_observation"], next_action)
                y_true = torch.clamp(y_true, min_value, max_value)

            y_pred = self._critic(batch["observation"], batch["action"])
            loss   = self._loss_function(y_pred, y_true)

            # Optim step
            self._critic_optimizer.zero_grad()
            loss.backward()
            self._critic_optimizer.step()

            # Update target network
            if ind % self._hyperparams["target_update_interval"] == 0:
                with torch.no_grad():
                    soft_sync(self._critic_targ, self._critic, tau=1.0 - self._hyperparams["polyak"])

            # Logging
            if self._logger:
                self._logger.log({
                    "counts/num_batches": ind + 1,
                    "counts/num_samples": (ind + 1) * self._hyperparams["batch_size"],
                    "loss/critic"       : loss
                })

            # Exit when enough updates are reached
            if ind >= int(self._hyperparams["n_gradient_steps"]):
                break

        # Save the latest model (just in case)
        self.save(self._hyperparams["policy_path"].replace(".policy", "_fqe.pt"))

    def eval(
        self,
        val_dataset: MDPDataset,
    ) -> Optional[float]:
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
                total      += torch.sum(self._critic(obs, policy_act))


        return (total / num_samples).cpu().item()


    def save(self, path: str) -> None:
        torch.save({
            "critic": self._critic.state_dict(),
            "critic_target": self._critic_targ.state_dict(),
            "critic_optimizer": self._critic_optimizer.state_dict(),
            "hyperparams": self._hyperparams
        }, path)

    def load(self, path: str, device: torch.device = "cpu") -> None:
        state_dict = torch.load(path, device = device)

        self._critic.load_state_dict(state_dict["critic"])
        self._critic_targ.load_state_dict(state_dict["critic_targ"])
        self._critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self._hyperparams = state_dict["hyperparams"]