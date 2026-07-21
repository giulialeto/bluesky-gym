import csv
import os
from stable_baselines3.common.callbacks import BaseCallback

EXCLUDED_INFO_KEYS = ('episode', 'terminal_observation', 'TimeLimit.truncated')

class CSVLoggerCallback(BaseCallback):
    """
    This callback saves the contents of the 'info' dictionary to a CSV file at the end of each episode.
    It also saves the SB3 training metrics to a separate CSV file.
    This logger supports parallelized environments. 
    """

    def __init__(self, log_dir, file_name='training_log.csv', verbose=0):
        super(CSVLoggerCallback, self).__init__(verbose)
        # check the directory exists, if not create it
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, file_name)

        base, ext = os.path.splitext(file_name)
        self.sb3_log_file = os.path.join(log_dir, f'{base}_SBlog{ext}')
        
        # Initialize the headers for the CSV file. The keys from the info dictionary are added later dynamically.
        self.headers = ['timesteps', 'episodes', 'env_idx']
        self.initialized = False

        #Log SB3 metrics
        self.sb3_initialized = False
        self.sb3_keys = []

        self.episode_count = 0

    def _on_step(self) -> bool:
        # Get the info from each of the parallelized environments. 'dones' specifies which environment has finished an episode.
        infos = self.locals['infos']
        dones = self.locals['dones']

        for env_idx, done in enumerate(dones):
            if done:
                # initialise the header of the CSV with dynamic keys from the info dictionary of the first episode that finishes running
                if not self.initialized:
                    self.info_keys = list(infos[env_idx].keys())
                    self.headers.extend(self.info_keys)
                    with open(self.log_file, mode='w', newline='') as f:
                        csv.writer(f).writerow(self.headers)
                    self.initialized = True
                
                self.episode_count += 1
                info_dict = infos[env_idx]
                # Writes the data from the finished episode to the CSV, excluding specific keys.
                row = [self.num_timesteps, self.episode_count, env_idx] + \
                      [info_dict.get(k, None) for k in self.info_keys if k not in EXCLUDED_INFO_KEYS]
                with open(self.log_file, mode='a', newline='') as f:
                    csv.writer(f).writerow(row)

                # SB3 training metrics — only populated after an update has occurred
                sb3_metrics = self.model.logger.name_to_value
                if sb3_metrics:
                    if not self.sb3_initialized:
                        self.sb3_keys = list(sb3_metrics.keys())
                        sb3_headers = ['timesteps', 'episodes'] + self.sb3_keys
                        with open(self.sb3_log_file, mode='w', newline='') as f:
                            csv.writer(f).writerow(sb3_headers)
                        self.sb3_initialized = True

                    sb3_values = [sb3_metrics.get(k, None) for k in self.sb3_keys]
                    sb3_row = [self.num_timesteps, self.episode_count] + sb3_values
                    with open(self.sb3_log_file, mode='a', newline='') as f:
                        csv.writer(f).writerow(sb3_row)

        return True