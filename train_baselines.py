import argparse
import os

from envs.core import create_env
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from utils.stable_baselines3.eval_callback import EvalSaveMultipleModelsCallback
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="industrial", choices=["finrl", "citylearn", "industrial"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_timesteps", type=int, default=int(1e6), help="How many (s,a,s',r) to train for.")
    parser.add_argument("--n_learning_starts", type=int, default=int(1e3))
    parser.add_argument("--eval_frequency", type=int, default=int(1e3))
    parser.add_argument("--eval_n_episodes", type=int, default=10)
    parser.add_argument("--sb3_path", type=str)
    args = parser.parse_args()

    # Some fixes for MacOS
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # create environment
    train_env = create_env(args.env)

    # configure saving paths
    save_path        = os.path.join(args.sb3_path, args.env)
    save_path_models = os.path.join(save_path, "best_models")
    configure(save_path, ["stdout"])

    # create a callback for saving best models
    eval_callback = EvalSaveMultipleModelsCallback(
        eval_env=create_env(args.env),
        best_models_save_path=save_path_models,
        n_eval_episodes=args.eval_n_episodes,
        eval_freq=args.eval_frequency,
        warn=False
    )

    # train
    model = SAC(
        policy="MlpPolicy", 
        env=train_env, 
        verbose=1, 
        learning_starts=args.n_learning_starts,
        batch_size=256,
        tensorboard_log=save_path,
        device=args.device
    )
    model.learn(
        total_timesteps=args.n_timesteps, 
        log_interval=4,
        callback=eval_callback
    )
