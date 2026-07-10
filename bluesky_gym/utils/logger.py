import csv
import os
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_dir, file_name='training_log.csv', verbose=0):
        super(CSVLoggerCallback, self).__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, file_name)
        self.headers = ['timesteps', 'episodes']
        self.initialized = False
        self.episode_count = 0

        # ADD
        self.reached_buffer = deque(maxlen=100)
        self.crashed_buffer = deque(maxlen=100)

    def _on_step(self) -> bool:
        if not self.initialized:
            # Initialize headers based on keys in the infos dictionary
            self.info_keys = self.locals['infos'][0].keys()
            self.headers.extend(self.info_keys)
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            self.initialized = True

        if self.locals['dones'][0]:
            self.episode_count += 1
            timesteps = self.num_timesteps
            info_dict = self.locals['infos'][0]

            # ADD
            self.reached_buffer.append(info_dict.get('waypoint_reached', 0))
            self.crashed_buffer.append(info_dict.get('crashed', 0))
            self.logger.record('rollout/waypoint_reached_mean', np.mean(self.reached_buffer))
            self.logger.record('rollout/crashed_mean', np.mean(self.crashed_buffer))

            info_values = [info_dict.get(key, None) for key in self.info_keys]
            row = [timesteps, self.episode_count] + list(info_values)
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        return True