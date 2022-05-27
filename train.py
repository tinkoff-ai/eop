# General
#   - seed
#   - dataset
#   - device
#   -
#   - n_epochs
#   - batch_size
#   -
#   - learning rate
#   - optimizer
#   - dataset
#   - gamma?
#   - target update interval?
#   - normalization on/off

# CQL hypers
#   ref: NeoRL paper and their code (this is cql orig https://github.com/aviralkumar2907/CQL)
#   - alpha (how much conservsative)
#   - tau   (auto-tunes alpha?)
#   - variants (entropy regularizer, kl-divergence regularizer)
#   - aproximate max-backup ()
#   - more sac hypeparams??
#       ref: https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/torch/sac/sac.py#L21
#       - use_automatic_entropy_tuning=True
#       - soft_target_tau=1e-2
#       - reward_scale=1.0

# TD3 + BC hypers
#   ref: https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py
#   bc hypers
#       - alpha (also conservatism)
#   td3 hypers
#       - policy noise 
#       - noise clip
#       - delayed policy update (how often update policy in comparison to value function?)
#       - tau (target network update rate)
#       - shared policy/value encoder???

# FisherBRC (uses BC inside) hypers
#   ref: https://github.com/google-research/google-research/blob/master/fisher_brc/train_eval_offline.py
#   bc hypers
#       - bc_pretraining_steps
#       - mixture vs one gaussian
#       - num components
#   fisher hypers
#       - f_reg
#       - reward_bonus (as in CQL?)

# CRR hypers

# Minimalist approach -- 1e6 gradient steps = 256e6 timesteps
# NeoRL -- 3e5 gradient steps = 76e6 timesteps
# CQL -- 1e6 gradient steps = 256e6 timesteps
# Ours -- 123123 = 9e6 timesteps
import argparse
import os
from typing import Any, Dict
from math import floor
from utils.normalize import Normalizer

import wandb
import algs.core as algs

from datasets.core import load_datasets
from datasets.torch import InfiniteMDPTorchDataset
from torch.utils.data.dataloader import DataLoader
from algs.core import sample_hyperparams
from utils.utils import fix_random_seeds
from utils.loader import BatchPrefetchLoaderWrapper
from itertools   import product
from multiprocessing import Process, connection
from tqdm import tqdm


def run_training(
    env: str, 
    alg: str, 
    n_trajectories: int, 
    policy: str, 
    seed: int, 
    device: str,
    hyperparams: Dict[str, Any]
):
    # Fix the seeds
    fix_random_seeds(seed)

    # Construct an algorithm
    alg_trainer = algs.create_trainer(alg_name=alg, env_name=env, hyperparams=hyperparams)
    alg_trainer.to(device)

    # # Load the datasets
    train_dataset, _ = load_datasets(env_name=env, n_trajectories=n_trajectories, policy_level=policy)

    # Add a normalizer to the trainer if needed
    if "normalize_obs" in hyperparams and hyperparams["normalize_obs"]:
        normalizer = Normalizer({
            "mean": train_dataset.observations.mean(0, keepdims=True),
            "std" : train_dataset.observations.std(0, keepdims=True)
        }, device=device)
        alg_trainer.set_normalizer(normalizer)

    # # Create logger
    logger = wandb.init(project="offline-rl-baseline", dir=os.environ["WANDB_PATH"], reinit=True)
    logger.config.update(hyperparams)
    logger.config.update({
        "env": env, 
        "alg": alg, 
        "n_trajectories": n_trajectories, 
        "policy_level": policy, 
        "seed": seed,
        "device": device}
    )
    alg_trainer.set_logger(logger)

    # # Iterator over the training dataset
    train_iterator = BatchPrefetchLoaderWrapper(
        loader= DataLoader(
            dataset    = InfiniteMDPTorchDataset(train_dataset),
            batch_size = int(hyperparams["batch_size"])
        ),
        device=device,
        num_prefetches=100
    )

    # # How often to save
    save_each_iteration = floor(hyperparams["n_gradient_steps"] / 40)

    ind = 0
    for ind, batch in enumerate(iter(train_iterator)):
        # Train!!!
        alg_trainer.step(batch)

        # Checkpoint the model
        if ind % save_each_iteration == 0:
            alg_trainer.save(name=f"epoch-{floor(ind / save_each_iteration)}")

        # Exit when enough updates reached
        if ind >= int(hyperparams["n_gradient_steps"]):
            break

    logger.finish()


def get_training_process(job: Dict[str, Any], env_name: str, device: str) -> Process:
    return Process(
            target=run_training, 
            kwargs={
                "env"           : env_name,
                "alg"           : job["alg"],
                "n_trajectories": job["n_trajectories"],
                "policy"        : job["policy"],
                "seed"          : job["seed"],
                "device"        : device,
                "hyperparams"   : job["hyperparams"]
            }
        )


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms",     type=str, nargs="+", default=["cql"])
    parser.add_argument("--env_name",       type=str, choices=["four-rooms", "finrl", "citylearn", "industrial"])
    parser.add_argument("--n_trajectories", type=int, nargs="+", default=[99, 999, 9999])
    parser.add_argument("--policies",       type=str, nargs="+", default=["medium"])
    parser.add_argument("--devices",        type=str, nargs="+", default=["cuda:0"])
    parser.add_argument("--n_hyperparams",  type=int, default=10, help="How many hyperparam assignments to sample per a setup.")
    parser.add_argument("--seeds",          type=int, nargs="+", default=[1712])
    args = parser.parse_args()

    # Some fixes for MacOS and wandb logging
    os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
    os.environ["WANDB_SILENT"]         = "True"

    # Fix the seeds
    fix_random_seeds(args.seeds[0])

    # Debug
    # if True:
    #     hyps = sample_hyperparams(env_name=args.env_name, alg_name=args.algorithms[0])
    #     # hyps["gmm_num_modes"] = 1
    #     run_training(
    #         env=args.env_name,
    #         alg=args.algorithms[0],
    #         n_trajectories=args.n_trajectories[0],
    #         policy=args.policies[0],
    #         seed=args.seeds[0],
    #         device=args.devices[0],
    #         hyperparams=hyps
    #     )
    #     exit(0)

    # Generate all the jobs
    prelim_jobs = product(args.seeds, args.algorithms, args.policies, args.n_trajectories)
    all_jobs    = []
    for (seed, alg, policy_level, n_trajectories) in prelim_jobs:
        for hyperparams in [sample_hyperparams(args.env_name, alg) for _ in range(args.n_hyperparams)]:
            all_jobs.append({
                "seed"          : seed,
                "alg"           : alg,
                "policy"        : policy_level,
                "n_trajectories": n_trajectories,
                "hyperparams"   : hyperparams
            })

    print(f"{len(all_jobs)} jobs to be run on {len(args.devices)} devices.")

    # Run training
    n_workers  = len(args.devices)
    pool       = [(get_training_process(job, args.env_name, device), device) for job, device in zip(all_jobs, args.devices)]
    n_jobs_run = len(args.devices)

    # Run the first batch of jobs
    for process, _ in pool:
        process.start()

    # Progress bar
    pbar = tqdm(total=len(all_jobs))

    # Run all others
    while n_jobs_run < len(all_jobs):
        # Wait until one of the processes is over
        connection.wait(process.sentinel for process, _ in pool)

        # Check which of the processes has terminated and released its device
        for ind in range(len(pool)):
            cur_device  = pool[ind][1]
            cur_process = pool[ind][0]

            # The job has been completed -- run a new one if needed
            if not cur_process.is_alive() and n_jobs_run < len(all_jobs):
                # Close this process
                cur_process.join()

                # Run a new one
                pool[ind] = (get_training_process(all_jobs[n_jobs_run], env_name=args.env_name, device=cur_device), cur_device)
                pool[ind][0].start()

                n_jobs_run += 1
                pbar.update(n=1)

    # Clean up
    for process, _ in pool:
        process.join()
        pbar.update(1)

    pbar.close()

