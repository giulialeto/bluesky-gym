import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

class SaveModelCallback(BaseCallback):
    """
    Callback for saving a model and its replay buffer at regular intervals during training.
    """

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save = 0

    def _init_callback(self) -> None:
        self.save_path = os.path.join(self.save_path, "checkpoint")
        # Create the irectory if it does not exist
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """ 
        Saves, with frequancy defined by save_freq, the model and its replay buffer 
        (only fot off-policy algorithms). Saves a tmp compy first, then replaces the 
        previous model and buffer with the new ones. This is to avoid corrupting the 
        model and buffer if the training is interrupted during saving."""
        
        if self.num_timesteps - self.last_save >= self.save_freq:
            model_path = os.path.join(self.save_path, f"model_latest.zip")
            tmp_path = os.path.join(self.save_path, f"model_latest_tmp.zip")

            self.model.save(tmp_path)      # write to temp file first
            os.replace(tmp_path, model_path)  # overwrite real file

            # off-policy algorithms (SAC/TD3/DDPG): also checkpoint the replay buffer
            if isinstance(self.model, OffPolicyAlgorithm):
                buffer_path = os.path.join(self.save_path, "model_latest_buffer.pkl")
                tmp_buffer_path = os.path.join(self.save_path, "model_latest_buffer_tmp.pkl")

                self.model.save_replay_buffer(tmp_buffer_path)
                os.replace(tmp_buffer_path, buffer_path)

            self.last_save = self.num_timesteps
            if self.verbose > 0:
                print(f"Model and buffer saved at step {self.num_timesteps} to {self.save_path}")

        return True