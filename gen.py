import os
import numpy as np
import argparse
import tqdm
import torch

from pathlib import Path
from policies.policy import Policy
from utils.utils  import fix_random_seeds, append_to_dataset
from utils.vec_env.subproc_vec_env import VecEnv, SubprocVecEnv
from datasets.dataset import MDPDataset
from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from envs.core import create_env, get_baseline
from copy import deepcopy


def generate_dataset(policy: Policy, env: VecEnv, n_trajectories: int = 99) -> MDPDataset:
    states                               = env.reset()
    n_envs                               = len(states)
    n_collected                          = 0
    env_data: Dict[int, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    dataset: Optional[MDPDataset]        = None

    # Progress bar
    pbar = tqdm.tqdm(total=n_trajectories)

    # Temporary data holders
    all_observations = []
    all_actions      = []
    all_rewards      = []
    all_terminals    = []

    while n_collected < n_trajectories:
        actions = policy.predict_actions(states)
        nstates, rewards, dones, infos = env.step(actions)

        # Log
        for env_ind in range(n_envs):
            terminated = "terminal_observation" in infos[env_ind]
            
            env_data[env_ind]["observations"].append(states[env_ind])
            env_data[env_ind]["actions"].append(actions[env_ind])
            env_data[env_ind]["rewards"].append(rewards[env_ind])
            env_data[env_ind]["terminals"].append(dones[env_ind])

            if terminated:
                # Push the trajectory to the MDP dataset
                all_observations.append(np.array(env_data[env_ind]["observations"]))
                # todo: potential bug where environments have only one action (it must be reshaped somehow)
                all_actions.append(np.array(env_data[env_ind]["actions"]))
                all_rewards.append(np.array(env_data[env_ind]["rewards"]).reshape(-1, 1))
                all_terminals.append(np.array(env_data[env_ind]["terminals"]).reshape(-1, 1))

                # Clean the trajectory
                env_data[env_ind] = defaultdict(list)

                # +1 trajectory
                n_collected += 1
                
                pbar.update(n=1)

        # Update states
        states = nstates

    # Obtain MDPDataset
    dataset = append_to_dataset(
        dataset=dataset,
        observations=np.vstack(all_observations),
        actions=np.vstack(all_actions),
        rewards=np.vstack(all_rewards),
        terminals=np.vstack(all_terminals),
    )
        
    # Close progress bard
    pbar.close()

    return dataset

def subsample_mdp_dataset(dataset: MDPDataset, n_episodes: int) -> MDPDataset:
    if len(dataset.episodes) == 0:
        raise Exception()
    if len(dataset.episodes) < n_episodes:
        raise Exception()

    # Return the same dataset (no copy!)
    if len(dataset.episodes) == n_episodes:
        return dataset

    indices = np.random.choice(a=range(len(dataset.episodes)), size=n_episodes, replace=False)
    data    = {
        "obs": [],
        "act": [],
        "rew": [],
        "ter": []
    }
    for ind in indices:
        # Need to reconstruct terminals
        terminals     = np.zeros(len(dataset.episodes[ind]))
        terminals[-1] = 1.0
        terminals = terminals.reshape(-1, 1)

        data["obs"].append(dataset.episodes[ind].observations)
        data["act"].append(dataset.episodes[ind].actions)
        data["rew"].append(dataset.episodes[ind].rewards.reshape(-1, 1))
        data["ter"].append(terminals)

    new_dataset = append_to_dataset(
        dataset=None,
        observations=np.vstack(data["obs"]),
        actions=np.vstack(data["act"]),
        rewards=np.vstack(data["rew"]),
        terminals=np.vstack(data["ter"])
    )

    return new_dataset

def split_mdp_dataset(dataset: MDPDataset, val: float) -> Tuple[MDPDataset, MDPDataset]:
    val_size = int(len(dataset.episodes) * val)
    if val_size >= len(dataset.episodes) or val_size <= 0:
        raise Exception()

    val_indices   = set(np.random.choice(a=range(len(dataset.episodes)), size=val_size, replace=False))

    train_data = {
        "obs": [],
        "act": [],
        "rew": [],
        "ter": []
    }
    val_data = deepcopy(train_data)

    for ind in range(len(dataset.episodes)):
        # Need to reconstruct terminals
        terminals     = np.zeros(len(dataset.episodes[ind]))
        terminals[-1] = 1.0
        terminals = terminals.reshape(-1, 1)

        if ind in val_indices:
            val_data["obs"].append(dataset.episodes[ind].observations)
            val_data["act"].append(dataset.episodes[ind].actions)
            val_data["rew"].append(dataset.episodes[ind].rewards.reshape(-1, 1))
            val_data["ter"].append(terminals)
        else:
            train_data["obs"].append(dataset.episodes[ind].observations)
            train_data["act"].append(dataset.episodes[ind].actions)
            train_data["rew"].append(dataset.episodes[ind].rewards.reshape(-1, 1))
            train_data["ter"].append(terminals)

    # Build datasets
    train_dataset = append_to_dataset(
        dataset=None,
        observations=np.vstack(train_data["obs"]),
        actions=np.vstack(train_data["act"]),
        rewards=np.vstack(train_data["rew"]),
        terminals=np.vstack(train_data["ter"])
    )
    val_dataset = append_to_dataset(
        dataset=None,
        observations=np.vstack(val_data["obs"]),
        actions=np.vstack(val_data["act"]),
        rewards=np.vstack(val_data["rew"]),
        terminals=np.vstack(val_data["ter"])
    )

    return train_dataset, val_dataset
    

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_folder",type=str)
    parser.add_argument("--env_name",       type=str, choices=["four-rooms", "finrl", "citylearn", "industrial"])
    parser.add_argument("--n_trajectories", type=int, nargs="+", default=[99, 999, 9999])
    parser.add_argument("--val",            type=float, default=0.1, help="How many trajectoreis are used for evaluation.")
    parser.add_argument("--policies",       type=str, nargs="+", default=["medium"])
    parser.add_argument("--n_workers",      type=int, default=3)
    parser.add_argument("--seed",           type=int, default=1712)
    parser.add_argument("--device",         type=str, default="cpu")
    args = parser.parse_args()

    # Some fixes for MacOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

    # Set the seed
    fix_random_seeds(seed=args.seed)

    # Create an environment
    env = SubprocVecEnv(env_fns=[lambda: create_env(args.env_name) for _ in range(args.n_workers)])
    env.seed(seed=args.seed)

    # Start collecting dataset for each baseline policy
    for level in args.policies:
        # Retrieve policy and put it on the target device
        policy  = get_baseline(args.env_name, level = level)
        policy.to(args.device)

        # Collec the dataset!
        dataset = generate_dataset(policy, env, n_trajectories=max(args.n_trajectories))

        # Print some stats
        stats = dataset.compute_stats()
        print(f"Policy: {level}")
        print(f"Return; Mean: {stats['return']['mean']}; Std: {stats['return']['std']}; Min: {stats['return']['min']}; Max: {stats['return']['max']};")
        print(f"Reward; Mean: {stats['reward']['mean']}; Std: {stats['reward']['std']}; Min: {stats['reward']['min']}; Max: {stats['reward']['max']};")

        # Split into smaller datasets
        for n_trajs in args.n_trajectories:
            dataset_n_trajs = subsample_mdp_dataset(dataset, n_episodes=n_trajs)

            # Save
            dataset_path = Path(os.path.join(args.datasets_folder, args.env_name))
            dataset_path.mkdir(parents=True, exist_ok=True)

            train, val = split_mdp_dataset(dataset_n_trajs, val=args.val)
            train.dump(os.path.join(dataset_path, f"{level}-{n_trajs}-train.h5"))
            val.dump(os.path.join(dataset_path, f"{level}-{n_trajs}-val.h5"))
