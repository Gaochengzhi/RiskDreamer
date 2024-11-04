from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from run_env import Highway_env

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from itertools import product
import concurrent.futures

# Define parameter combinations
param_guess = {"a": [1], "b": [100, 10, 5, 20], "c": [0.1, 1, 10]}


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0, name=""):
        super(CustomCallback, self).__init__(verbose)
        self.reset_count = 0
        self.should_stop = False
        self.name = name

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        if env.reset_times != self.reset_count:
            self.reset_count = env.reset_times
            # if self.reset_count % 500 == 0:
            #     self.model.save(
            #         f"../model/{self.name}{self.reset_count}")
            if self.reset_count >= 2000:
                self.should_stop = True
                return False
        return True


def run_one_try(params):
    a, b, c = params
    name = f"sac_{params}"
    wandb.init(project="compare", name=name)
    env = Highway_env(param=[a, b, c], gui=False)
    model = SAC("MlpPolicy", env, verbose=1)
    custom_callback = CustomCallback()
    model.learn(
        total_timesteps=4000 * 10 * 400,
        callback=custom_callback
    )
    model.save(f"sac_model_{params}")


combinations = list(
    product(param_guess["a"], param_guess["b"], param_guess["c"]))


def parallel_run():
    # We use max_workers=22 to use 22 CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=18) as executor:
        futures = [executor.submit(run_one_try, params)
                   for params in combinations]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will re-raise exceptions if they occurred
            except Exception as e:
                print(f"Exception occurred: {e}")


if __name__ == "__main__":
    parallel_run()
    # run_one_try([1, 1, 0.1])
