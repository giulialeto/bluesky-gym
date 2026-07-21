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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import datetime
import os
import numpy as np
import bluesky_gym
import traceback
import shutil

from bluesky_gym.utils import logger_parallelised_callback as logger
from bluesky_gym.utils import model_checkpoint_callback

bluesky_gym.register_envs()

env_name = 'StaticObstacleCREnv-v1'
algorithm = PPO

num_cpu = 7
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{str(algorithm.__name__)}.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)
    
TRAIN = False
EVAL_EPISODES = 10

resume_from_checkpoint = False  # Set to True to resume training from the last checkpoint

# Initialise the environment counter
env_counter = 0

if __name__ == "__main__":

    # training and evaluation
    tic = datetime.datetime.now()
    env = make_vec_env(env_name,
            n_envs=num_cpu,
            seed=0,
            vec_env_cls=SubprocVecEnv)

    save_path=(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/")

    save_callback = model_checkpoint_callback.SaveModelCallback(save_freq=50000, save_path=save_path, verbose=1)

    checkpoint_dir = os.path.join(save_path, "checkpoint")
    model_path = os.path.join(checkpoint_dir, "model_latest.zip")
    buffer_path = os.path.join(checkpoint_dir, "model_latest_buffer.pkl")

    if os.path.exists(model_path) and resume_from_checkpoint:
        print(f"Resuming from checkpoint: {model_path}")
        model = algorithm.load(model_path, env=env)
        if os.path.exists(buffer_path):
            model.load_replay_buffer(buffer_path)
            print("Replay buffer restored")
        resumed = True
    else:
        model = algorithm("MultiInputPolicy", env, verbose=1, learning_rate=3e-4)
        resumed = False

    if TRAIN:
        remaining_timesteps = 1e7 - model.num_timesteps if resumed else 1e7

        model.learn(total_timesteps=remaining_timesteps, callback = CallbackList([save_callback,csv_logger_callback]), reset_num_timesteps=not resumed)
        model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model")
        del model
        # final model saved successfully, clean up intermediate checkpoints
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    env.close()
    del env

    # Test the trained model
    env = gym.make(env_name, render_mode="human")
    model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model", env=env)
    for i in range(EVAL_EPISODES):

        done = truncated = False
        obs, info = env.reset()
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