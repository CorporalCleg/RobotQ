import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import numpy as np


# template of environment
class OurAwesomeEnv(gym.env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):  # add other parameters

        self.observaition_space = (
            spaces.Box()
        )  # https://gymnasium.farama.org/api/spaces/
        self.action_space = spaces.Box()

    def reset(self, seed=0) -> int:  # reset environment
        super().reset(seed=seed)
        pass

    def step(self, action: int) -> (
        int,
        float,
        bool,
        bool,
        Optional[str],
    ):  # (new_state, reward, terminated, truncated, info)
        pass

    def close(self):
        pass

    def render(self) -> np.ndarray:
        pass
