import csv
import os
from stable_baselines3.common.callbacks import BaseCallback

class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_dir, file_name='training_log.csv', verbose=0):
        super(CSVLoggerCallback, self).__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, file_name)

        base, ext = os.path.splitext(file_name)
        self.sb3_log_file = os.path.join(log_dir, f'{base}_SBlog{ext}')

        self.headers = ['timesteps', 'episodes']
        self.initialized = False

        #Log SB3 metrics
        self.sb3_initialized = False
        self.sb3_keys = []

        self.episode_count = 0

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
            info_values = [info_dict.get(key, None) for key in self.info_keys]
            row = [timesteps, self.episode_count] + list(info_values)
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

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
                sb3_row = [timesteps, self.episode_count] + sb3_values
                with open(self.sb3_log_file, mode='a', newline='') as f:
                    csv.writer(f).writerow(sb3_row)


        return True