"""
This file is an example train and test loop for the different environments.
Selecting different environments is done through setting the 'env_name' variable.

TODO:
* add rgb_array rendering for the different environments to allow saving videos
"""

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import datetime
import os
import numpy as np
import bluesky_gym
import shutil

from bluesky_gym.utils import logger_parallelised_callback as logger
from bluesky_gym.utils import model_checkpoint_callback
from bluesky_gym.utils.seeding import derive_seed

bluesky_gym.register_envs()

env_name = 'StaticObstacleCREnv-v1'
algorithm = SAC

num_cpu = 7
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Rendering: with RENDER = True every training worker opens its own pygame window ("human")
RENDER = False
RENDER_COLUMNS = 2      # windows per row;
WINDOW_SIZE = 512       # pixels, the size of the window the environments open
WINDOW_MARGIN = 10      # pixels between windows
WINDOW_TOP = 20         # pixels, keeps the first row below the menu bar
WINDOW_TITLE_BAR = 30   # pixels, extra height taken by each window's title bar

TRAIN = True
EVAL_EPISODES = 10
TIMESTEPS = 4000003 #1e7

# Base seed: every other seed (policy, one per training worker, evaluation episodes) is derived from it,
# so the same SEED reproduces the same training run and the same evaluation episodes.
SEED = 42

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{str(algorithm.__name__)}_{TIMESTEPS}.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

resume_from_checkpoint = False  # Set to True to resume training from the last checkpoint

# Initialise the environment counter
env_counter = 0

def make_env(rank, env_name, render, columns):
    """Environment per worker. With render=True the worker's window gets its own grid position."""
    col, row = rank % columns, rank // columns
    window_pos = (col * (WINDOW_SIZE + WINDOW_MARGIN),
                  WINDOW_TOP + row * (WINDOW_SIZE + WINDOW_TITLE_BAR + WINDOW_MARGIN))

    def _init():
        if render:
            # Each worker is a separate process. The environment creates its window at the first render
            # and SDL reads the position at that moment, so it is set here, before the environment is built.
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{window_pos[0]},{window_pos[1]}"
        return Monitor(gym.make(env_name, render_mode="human" if render else None))
    return _init


if __name__ == "__main__":

     # training and evaluation
    tic = datetime.datetime.now()

    # define the environment with parallelized environments using SubprocVecEnv
    if RENDER:
        env = SubprocVecEnv([make_env(i, env_name, True, RENDER_COLUMNS) for i in range(num_cpu)], start_method="spawn")
        env.seed(derive_seed(SEED, "env"))  # worker i gets this seed + i, like make_vec_env(seed=...)
    else:
        env = make_vec_env(env_name,
                n_envs=num_cpu,
                seed=derive_seed(SEED, "env"),  # worker i gets this seed + i
                vec_env_cls=SubprocVecEnv)
   
    # env = gym.make(env_name, render_mode=None, max_episode_steps=200)

    save_path=(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/")
    save_callback = model_checkpoint_callback.SaveModelCallback(save_freq=50000, save_path=save_path, verbose=1)

    # define the path and frequency for saving checkpoints of the model and its replay buffer
    checkpoint_dir = os.path.join(save_path, "checkpoint")
    model_path = os.path.join(checkpoint_dir, "model_latest.zip")
    buffer_path = os.path.join(checkpoint_dir, "model_latest_buffer.pkl")

    # define the model (either anew or from a checkpoint)
    if os.path.exists(model_path) and resume_from_checkpoint:
        print(f"Resuming from checkpoint: {model_path}")
        model = algorithm.load(model_path, env=env)
        # do not replay the episodes of the first run: new env seed tied to the resume point
        # the checkpoint does not store the state of the random number generator, so the model trained on the resumed env will be different from the model trained in one go. It is possible to reproduce the training starting for the same checkpoint by means of the same seed here
        env.seed(derive_seed(SEED, "env", "resume", model.num_timesteps))
        if os.path.exists(buffer_path):
            model.load_replay_buffer(buffer_path)
            print("Replay buffer restored")
        resumed = True
    else:
        model = algorithm("MultiInputPolicy", env, verbose=1, learning_rate=3e-4, seed=derive_seed(SEED, "model"), learning_starts=10000)
        # SB3 seeds the workers with the model seed (+ index); give them their own derived seed instead
        env.seed(derive_seed(SEED, "env"))
        resumed = False

    # model = algorithm("MultiInputPolicy", env, verbose=1, learning_rate=3e-4, seed=42, learning_starts=10000)
    
    
    if TRAIN:
        remaining_timesteps = TIMESTEPS - model.num_timesteps if resumed else TIMESTEPS

        model.learn(total_timesteps=remaining_timesteps, callback = CallbackList([save_callback,csv_logger_callback]), reset_num_timesteps=not resumed)
        model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_{TIMESTEPS}")
        del model
        # final model saved successfully, clean up intermediate checkpoints
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    # if TRAIN:
    #     model.learn(total_timesteps=TIMESTEPS, callback=csv_logger_callback)
    #     model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_{TIMESTEPS}")
    #     del model
    env.close()
    del env

    # Test the trained model
    env = gym.make(env_name, render_mode="human")
    model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_{TIMESTEPS}", env=env)
    for i in range(EVAL_EPISODES):
        done = truncated = False
        obs, info = env.reset(seed=derive_seed(SEED, "eval", i))
        tot_rew = 0
        while not (done or truncated):
            # action = np.array(np.random.randint(-100,100,size=(2))/100)
            # action = np.array([0,-1])
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action[()])
            tot_rew += reward
        print(tot_rew)
    env.close()

    toc = datetime.datetime.now()
    print(f'Finished evaluation at {toc}')
    print(f'Elapsed time: {toc - tic}')    