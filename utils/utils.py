import random
import torch
import numpy as np

from datasets.dataset import MDPDataset
from typing           import Optional

def fix_random_seeds(seed: int) -> None:
    """
    Use in the beginning of the program only.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    np.random.seed(seed)
    random.seed(seed)


def append_to_dataset(
    dataset: Optional[MDPDataset], observations: np.ndarray, 
    actions: np.ndarray, rewards: np.ndarray, terminals: np.ndarray
) -> MDPDataset:
    if dataset is None:
        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals
        )
    else:
        dataset.append(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals
        )

    return dataset