from collections import defaultdict
import gym
import numpy as np
import argparse
import os
import torch
import wandb

from typing import Any, Tuple, Dict, List
from utils.vec_env.subproc_vec_env import VecEnv, SubprocVecEnv
from policies.policy import Policy
from pathlib import Path
from utils.wandb import get_run_from_folder, get_policy_paths_from_folder
from multiprocessing import Process, connection, Queue
from envs.core import create_env
from tqdm import tqdm
from algs.core import load_alg_policy


def eval_online(
    policy_path: Path, 
    env_name: str,
    alg_name: str,
    device: torch.device,
    result_holder: Queue,
    n_workers: int = 4, 
    n_trajectories: int = 1000
) -> float:
    # Load the policy
    policy = load_alg_policy(alg_name=alg_name, env_name=env_name, policy_path=policy_path)
    policy.to(device)

    # Create vectorized environment
    env = SubprocVecEnv(env_fns=[lambda: create_env(env_name) for _ in range(n_workers)])

    # Start collecting trajectories and computing vs0
    states                               = env.reset()
    n_envs                               = len(states)
    n_collected                          = 0
    env_rewards: Dict[int, List[float]]  = defaultdict(list)
    s0_values                            = []

    while n_collected < n_trajectories:
        actions = policy.predict_actions(states)
        nstates, rewards, _, infos = env.step(actions)

        # Log
        for env_ind in range(n_envs):
            terminated = "terminal_observation" in infos[env_ind]
            env_rewards[env_ind].append(rewards[env_ind])

            if terminated:
                n_collected += 1
                s0_values.append(np.sum(env_rewards[env_ind]))
                env_rewards[env_ind] = []

        # Update states
        states = nstates

    # Clean up
    env.close()
        
    # Put on a result
    result_holder.put(np.mean(s0_values))


def get_evaluation_process(
    job: Dict[str, Any], 
    device: torch.device
) -> Tuple[Process, Queue]:
    result_holder = Queue()
    return Process(
            target=eval_online, 
            kwargs={
                "policy_path"   : job["policy_path"],
                "env_name"      : job["env_name"],
                "alg_name"      : job["alg"],
                "n_trajectories": job["n_trajectories"],
                "n_workers"     : job["n_workers"],
                "device"        : device,
                "result_holder" : result_holder
            }
        ), result_holder


DEFAULT_N_TRAJECTORIES = {
    "industrial": 100,
    "finrl"     : 5,
    "citylearn" : 200
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs",      type=str, nargs="+", default=["four-room", "industrial", "citylearn", "finrl"], help="For which environments start online evaluation.")
    parser.add_argument("--algs",      type=str, nargs="+", default=["cql", "td3+bc", "bc", "fisherbrc", "crr"])
    parser.add_argument("--levels",    type=str, nargs="+", default=["low", "medium", "high"])
    parser.add_argument("--n_workers", type=int, default=4, help="How many environments per device.")
    parser.add_argument("--devices",   type=str, nargs="+", default=["cpu"])
    parser.add_argument("--n_trajectories", type=int, default=1000, help="How many rollouts per policy.")
    parser.add_argument("--strategy", type=str, default="40", choices=["last", "all", "40"])
    parser.add_argument("--wandb_entity", type=str, default="vkurenkov")
    parser.add_argument("--wandb_project_name", type=str, default="offline-rl-baseline")
    parser.add_argument("--force", default=False, type=bool)
    args = parser.parse_args()

    # Some fixes for MacOS and wandb logging
    os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
    os.environ["WANDB_SILENT"]         = "True"

    ### Parse all the wandb runs to prepare evaluation jobs ###
    eval_jobs  = []
    wandb_path = Path(os.path.join(os.environ["WANDB_PATH"], "wandb"))
    subfolders = [f for f in os.scandir(wandb_path) if f.is_dir()]
    wandb_api  = wandb.Api()
    for folder in tqdm(subfolders, desc="Parsing local wandb folders"):
        # Skip latest-run folder
        if folder.name == "latest-run":
            continue
        
        # Parse wandb run from this folder
        run = get_run_from_folder(
            folder_name=folder.name, 
            entity=args.wandb_entity, 
            project_name=args.wandb_project_name,
            wandb_api=wandb_api
        )

        # This run is not on the sever thus skip
        if not run:
            continue

        # If not in the target envs -- skip
        if "env" not in run.config or run.config["env"] not in args.envs:
            continue
        if "policy_level" not in run.config or run.config["policy_level"] not in args.levels:
            continue
        if "alg" not in run.config or run.config["alg"] not in args.algs:
            continue

        # If the training is still running or crushed -- skip
        if run.state != "finished":
            continue

        # Extract paths to each policy and add jobs
        policies = get_policy_paths_from_folder(folder_path=folder.path)
        # Only if the last policy needs to be evaluated
        if args.strategy == "last" or args.strategy == "40":
            policies = [policies[-1]]

        # Add evaluation jobs
        for (policy_path, epoch) in policies:
            # If the target performance metric is already calculated -- skip
            key_name = f"epoch_{epoch}/online_vs0"
            if key_name in run.summary and not args.force:
                continue

            # Only 40th epoch....
            if args.strategy == "40" and epoch != 40:
                continue

            eval_jobs.append({
                "policy_path"   : policy_path,
                "epoch"         : epoch,
                "env_name"      : run.config["env"],
                "n_trajectories": DEFAULT_N_TRAJECTORIES[run.config["env"]],
                "n_workers"     : args.n_workers,
                "run"           : run,
                "alg"           : run.config["alg"]
            })

    print(f"{len(eval_jobs)} jobs to be run on {len(args.devices)} devices.")

    ### Run the evaluation jobs ###
    n_workers  = len(args.devices)
    pool       = []
    for job, device in zip(eval_jobs, args.devices):
        eval_prc, eval_res = get_evaluation_process(job=job, device=device)
        pool.append((eval_prc, eval_res, device, job))
    n_jobs_run = len(args.devices)

    # Run the first batch of jobs
    for process, _, _, _ in pool:
        process.start()

    # Progress bar
    pbar = tqdm(total=len(eval_jobs), desc="Running evaluation jobs")

    # Run all others
    while n_jobs_run < len(eval_jobs):
        # Wait until one of the processes is over
        connection.wait(process.sentinel for process, _, _ , _ in pool)

        # Check which of the processes has terminated and released its device
        for ind in range(len(pool)):
            cur_process = pool[ind][0]
            cur_res     = pool[ind][1]
            cur_device  = pool[ind][2]
            cur_job     = pool[ind][3]

            # The job has been completed -- run a new one if needed
            if not cur_process.is_alive() and n_jobs_run < len(eval_jobs):
                # Retrieve the result
                vs0 = cur_res.get()
                # Close this process
                cur_process.join()

                # Log the result into wandb
                cur_job["run"].summary[f"epoch_{cur_job['epoch']}/online_vs0"] = vs0
                cur_job["run"].summary.update()

                # Run a new one
                new_job = eval_jobs[n_jobs_run]
                eval_prc, eval_res = get_evaluation_process(job=new_job, device=cur_device)
                pool[ind] = (eval_prc, eval_res, cur_device, new_job)
                pool[ind][0].start()

                n_jobs_run += 1
                pbar.update(n=1)

    # Clean up
    for process, cur_res, _, cur_job in pool:
        # Retrieve the result
        vs0 = cur_res.get()
        # Close this process
        process.join()

        # Log the result into wandb
        cur_job["run"].summary[f"epoch_{cur_job['epoch']}/online_vs0"] = vs0
        cur_job["run"].summary.update()

        pbar.update(1)


