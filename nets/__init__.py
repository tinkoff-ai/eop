import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math
import numpy as np

from utils.distributions import TanhNormal
from utils.torch import soft_clamp
from typing import Tuple


class ValueMLPNet(nn.Module):
	def __init__(self, state_dim: int, action_dim: int):
		super(ValueMLPNet, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		return q1


class TwoHeadValueMLPNet(nn.Module):
	def __init__(
        self, 
        state_dim: int,
        action_dim: int
    ):
		super(TwoHeadValueMLPNet, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(
        self, 
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TanhGaussianMLPNet(nn.Module):
    LOG_SIG_MAX: float = 2
    LOG_SIG_MIN: float = -5
    MEAN_MIN   : float = -9.0
    MEAN_MAX   : float  = 9.0

    def __init__(
        self,
        obs_dim: int,
        action_dim: int
    ):
        super(TanhGaussianMLPNet, self).__init__()

        self._input_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self._mu = nn.Linear(256, action_dim)
        self._sigma = nn.Linear(256, action_dim)

    def forward(
        self,
        observation: torch.Tensor
    ) -> Tuple[TanhNormal, torch.Tensor, torch.Tensor]:
        """
        :param obs: Observation
        """

        # Compute ouput distribution
        hidden = self._input_encoder(observation)
        mean   = torch.clamp(
            self._mu(hidden),
            min=self.MEAN_MIN,
            max=self.MEAN_MAX
        )
        std    = torch.clamp(
            self._sigma(hidden), 
            min=self.LOG_SIG_MIN, 
            max=self.LOG_SIG_MAX
        ).exp()
        final_distr = TanhNormal(mean, std)

        # Sample actions and compute its log prob
        sampled_actions, pretanh_value = final_distr.rsample(return_pretanh_value=True)
        sampled_logprobs = final_distr.log_prob(sampled_actions, pretanh_value).sum(-1)

        return final_distr, sampled_actions, sampled_logprobs


class TanhClampedMLPNet(nn.Module):
	def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        max_action: float
    ):
		super(TanhClampedMLPNet, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state: torch.Tensor) -> torch.Tensor:
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class GMMMlpNet(nn.Module):
    LOG_SIG_MAX: float = 1.0
    LOG_SIG_MIN: float = -10
    MEAN_MIN   : float = -9.0
    MEAN_MAX   : float = 9.0

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_modes: int
    ):
        super(GMMMlpNet, self).__init__()

        self._action_dim = action_dim
        self._obs_dim = obs_dim
        
        self._num_modes = num_modes
        self._input_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self._mu = nn.Linear(256, action_dim * num_modes)
        self._sigma = nn.Linear(256, action_dim * num_modes)
        self._logits = nn.Linear(256, num_modes)

    def forward(
        self,
        observation: torch.Tensor,
        low_std: bool = False
    ) -> Tuple[TanhNormal, torch.Tensor, torch.Tensor]:
        """
        :param obs: Observation
        """
        encoded = self._input_encoder(observation)
        mean    = self._mu(encoded)
        log_std = self._sigma(encoded)

        # Simple gaussian (as in NeoRL)
        if self._num_modes == 1:
            # Clamp the mean
            mean = torch.clamp(mean, min=self.MEAN_MIN, max=self.MEAN_MAX)
            # Clamp the scale
            log_std = soft_clamp(log_std, self.LOG_SIG_MIN, self.LOG_SIG_MAX)

            return D.Normal(mean, torch.exp(log_std))
        # Gaussian Mixture (as in What Matters ...)
        else:
            logits  = self._logits(encoded)
            mean    = mean.view(mean.size(0), self._num_modes, -1)
            log_std = log_std.view(log_std.size(0), self._num_modes, -1)

            # Scale the std
            if low_std:
                # low-noise for all Gaussian dists
                log_std = torch.ones_like(mean) * 1e-4
            else:
                log_std = F.softplus(log_std) + math.exp(self.LOG_SIG_MIN)

            # mixture components - make sure that `batch_shape` for the distribution is equal
            # to (batch_size, num_modes) since MixtureSameFamily expects this shape
            component_distribution = D.Normal(loc=mean, scale=log_std)
            component_distribution = D.Independent(component_distribution, 1)

            # unnormalized logits to categorical distribution for mixing the modes
            mixture_distribution = D.Categorical(logits=logits)

            dist = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )

            return dist


class ValueConvNet(nn.Module):
    """
    The same architecture that worked using d3rlpy in FourRooms-v1.
    """
    
    def __init__(self, obs_shape: np.ndarray, n_actions: int):
        super(ValueConvNet, self).__init__()

        # Should be the same we used in d3rlpy
        self._conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        n_features = self._conv_encoder(torch.zeros(obs_shape).unsqueeze(0)).size(-1)

        self._value_head = nn.Sequential(
            nn.Linear(n_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_actions)
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self._value_head(
            self._conv_encoder(
                batch
            )
        )