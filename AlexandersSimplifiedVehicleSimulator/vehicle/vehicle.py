from gymnasium import spaces
import numpy as np

class Vehicle():
    """
    Base class for vehicles
    """
    def __init__(self, seed=None) -> None:
        # Action space for the otter is +-100%
        self.action_space = spaces.Box(low=-1.0,high=1.0,shape=(2,),tyoe=np.float32,seed=seed)

    def step():
        """
        Normal step method for simulation
        """
        pass

    def rl_step():
        """
        Step method for RL purposes
        """
        pass