import gymnasium as gym
import numpy as np

class NoisyObservationWrapper(gym.Wrapper):
    def __init__(self, env, noise_level=0.1, seed=None):
        super().__init__(env)
        self.noise_level = noise_level
        self._seed_noise(seed)

    def _seed_noise(self, seed):
        # own generator, independent of the environment's one, so that adding noise
        # does not change the scenarios the environment samples
        entropy = None if seed is None else [int(seed), 1]
        self.noise_rng = np.random.default_rng(np.random.SeedSequence(entropy))

    def reset(self, **kwargs):
        if kwargs.get("seed") is not None:
            self._seed_noise(kwargs["seed"])
        observation, info = self.env.reset(**kwargs)
        noisy_observation = self.add_noise(observation)
        return noisy_observation, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        noisy_observation = self.add_noise(observation)
        return noisy_observation, reward, done, truncated, info

    def add_noise(self, observation):
        # Add Gaussian noise to the observation
        if isinstance(observation, np.ndarray):
            noise = self.noise_rng.normal(0, self.noise_level, size=observation.shape)
            return observation + noise
        elif isinstance(observation, dict):
            return {
                key: (value + self.noise_rng.normal(0, self.noise_level, size=value.shape) 
                      if isinstance(value, np.ndarray) else value)
                for key, value in observation.items()
                }
        else:
            print('observation not an numpy array or dictionary, return observation unaltered')
            return observation