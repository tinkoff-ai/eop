from pathlib import Path
from typing import List, Tuple, Optional, List
from wandb.wandb_run import Run
from wandb.apis import PublicApi

import wandb
import os


def get_run_from_folder(
    folder_name: str, 
    entity: str, 
    project_name: str, 
    wandb_api: PublicApi
) -> Optional[Run]:
    run_id = folder_name.split("-")[2]
    try:
        run = wandb_api.run(f"{entity}/{project_name}/{run_id}")
        return run
    except:
        return None


def get_policy_paths_from_folder(folder_path: str) -> List[Tuple[Path, int]]:
    # Check if files at least present
    files_path = Path(os.path.join(folder_path, "files"))
    if not files_path.exists():
        return [], []

    # Extract all policies
    policies = []
    for file in os.listdir(files_path):
        if file.endswith(".policy"):
            path  = os.path.join(files_path, file)
            epoch = int(file.replace(".policy", "").split("-")[1])
            policies.append((path, epoch))
    
    # Sort by epoch
    policies.sort(key=lambda x:x[1])
    
    return policies