import numpy as np

from stable_baselines3.common.base_class import BaseAlgorithm
from policies.policy import Policy
from typing import Any, List


class SB3BasedPolicy(Policy):
    def __init__(
        self,
        sb3_model: BaseAlgorithm
    ):
        self._sb3_model = sb3_model

    def predict_actions(self, obs: List[np.ndarray]) -> List[Any]:
        actions, _ = self._sb3_model.predict(
            observation=np.array(obs),
            deterministic=True
        )
        
        return actions