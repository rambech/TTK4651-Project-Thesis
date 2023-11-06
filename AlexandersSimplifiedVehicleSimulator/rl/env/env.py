"""
This is the base class for all my environments
"""

import gymnasium as gym
from gymnasium import spaces
from vehicle import Vehicle


class Env(gym.Env):
    """
    Base class for gym environments
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, vehicle: Vehicle) -> None:
        super(Env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = vehicle.action_space
        self.observation_space = spaces.Box(-1, 1, (2,))

    def step(self, action):
        pass

    def reset(self, seed=None, option=None):
        pass
