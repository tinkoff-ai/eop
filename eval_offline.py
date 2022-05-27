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
from tqdm import tqdm
from algs.core import load_alg_policy, load_trainer
from utils.utils import fix_random_seeds
from datasets.core import load_datasets
from ope.core import OPE_NAME_VAL_S0_FQE, create_eval_trainer, sample_hyperparams


# Done 1. Loss on validation data
#   - What Matters in 

# Done? 2. TD-Error on validation data
#   - ??

# 3. V_s0 using Q from ALG
#   - ??

# 4. V_s0 using Q from FQE
#   1. FQE from NeoRL (and Paine et al.) 
#   2. FQE from Kostrikov and Nachum


def eval_offline(
    wandb_config: Dict[str, Any],
    train_config: Dict[str, Any],
    device: torch.device,
    result_holder: Queue,
) -> float:
    # Fix the seeds
    fix_random_seeds(train_config["seed"])

    # Load the policy
    policy = load_alg_policy(
        alg_name    = wandb_config["alg"],
        env_name    = wandb_config["env"],
        policy_path = train_config["policy_path"]
    )
    policy.to(device)

    # Load the policy's trainer
    policy_trainer = load_trainer(
        alg_name    = wandb_config["alg"],
        env_name    = wandb_config["env"],
        hyperparams = wandb_config,
        path        = train_config["trainer_path"],
        device      = device
    )
    policy_trainer.to(device)

    # # Load the datasets
    train_dataset, val_dataset = load_datasets(
        env_name       = wandb_config["env"],
        n_trajectories = wandb_config["n_trajectories"],
        policy_level   = wandb_config["policy_level"],
        val_only       = True
    )
    
    # Create the evaluation trainer
    train_config.update(sample_hyperparams(train_config["ope_name"]))
    train_config.update({"env": wandb_config["env"]})
    eval_trainer = create_eval_trainer(
        alg_name       = train_config["ope_name"],
        policy_trainer = policy_trainer,
        policy         = policy,
        train_config   = train_config,
    )
    eval_trainer.to(device)

    # Set the logger if one uses FQE
    logger = None
    if train_config["ope_name"] == OPE_NAME_VAL_S0_FQE:
        logger = wandb.init(project="offline-rl-ope", dir=os.environ["WANDB_PATH"], reinit=True)
        logger.config.update(train_config)
        logger.config.update({"device": device})
    eval_trainer.set_logger(logger)

    # Train it (for some offline evals it's noop)
    # Offline algos typically use val_dataset only, so train_dataset is None for memory reasons
    eval_trainer.train(train_dataset, val_dataset)

    # Compute offline metric
    offline_metric = eval_trainer.eval(val_dataset)

    # Put on a result
    result_holder.put(offline_metric)

    # Finish the logger
    if logger:
        logger.finish()


def get_evaluation_process(
    job: Dict[str, Any], 
    device: torch.device
) -> Tuple[Process, Queue]:
    result_holder = Queue()
    return Process(
            target=eval_offline, 
            kwargs={
                "wandb_config" : job["wandb_config"],
                "train_config" : job["train_config"],
                "device"       : device,
                "result_holder": result_holder
            }
        ), result_holder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs",      type=str, nargs="+", default=["industrial", "finrl", "citylearn"], help="For which environments start offline evaluation.")
    parser.add_argument("--algs",      type=str, nargs="+", default=["cql", "td3+bc", "bc", "fisherbrc", "crr"])
    parser.add_argument("--levels",    type=str, nargs="+", default=["low", "medium", "high"])
    parser.add_argument("--devices",   type=str, nargs="+", default=["cpu"])
    parser.add_argument("--strategy", type=str, default="40", choices=["last", "all", "40"])
    parser.add_argument("--wandb_entity", type=str, default="vkurenkov")
    parser.add_argument("--wandb_project_name", type=str, default="offline-rl-baseline")
    parser.add_argument("--ope_name", 
        type=str, 
        default="val_vs0_fqe", 
        choices=["val_loss", "val_tderror", "val_vs0_q", "val_vs0_fqe"]
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", type=bool, default=False)
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
            key_name = f"epoch_{epoch}/{args.ope_name}"
            if key_name in run.summary and not args.force:
                continue

            # Only 40th epoch....
            if args.strategy == "40" and epoch != 40:
                continue

            eval_jobs.append({
                "run"         : run,
                "epoch"       : epoch,
                "wandb_config": run.config,
                "train_config": {
                    "trainer_path": str(policy_path).replace(".policy", ".pt"),
                    "policy_path": policy_path,
                    "ope_name": args.ope_name,
                    "seed": args.seed
                }
            })

    print(f"{len(eval_jobs)} jobs to be run on {len(args.devices)} devices.")

    # Debug
    # if True:
    #     job = eval_jobs[0]
    #     result_holder = Queue()
    #     eval_offline(
    #         wandb_config=job["wandb_config"],
    #         train_config=job["train_config"],
    #         device="cpu",
    #         result_holder=result_holder
    #     )
    #     print(result_holder.get())

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
                offline_metric = cur_res.get()
                # Close this process
                cur_process.join()

                # Log the result into wandb
                cur_job["run"].summary[f"epoch_{cur_job['epoch']}/{args.ope_name}"] = offline_metric
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
        offline_metric = cur_res.get()
        # Close this process
        process.join()

        # Log the result into wandb
        cur_job["run"].summary[f"epoch_{cur_job['epoch']}/{args.ope_name}"] = offline_metric
        cur_job["run"].summary.update()

        pbar.update(1)


