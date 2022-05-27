import os

from datasets.dataset import MDPDataset
from typing import Tuple, Optional
from pathlib import Path


def load_datasets(
    env_name      : str,
    n_trajectories: int,
    policy_level  : str,
    val_only      : bool = False
) -> Tuple[Optional[MDPDataset], Optional[MDPDataset]]:
    path       = os.path.join(os.environ["DATASETS_PATH"], env_name, f"{policy_level}-{n_trajectories}")
    path_train = Path(f"{path}-train.h5")
    path_val   = Path(f"{path}-val.h5")

    if not path_train.exists():
        raise Exception("There is no such dataset.")

    # Load train dataset
    train_dataset = None
    if not val_only:
        train_dataset = MDPDataset.load(path_train)

    # Load val dataset
    val_dataset   = None
    if path_val.exists():
        val_dataset = MDPDataset.load(path_val)
    
    return train_dataset, val_dataset

