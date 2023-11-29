"""
Main script for running Vehicle simulator

"""
import gym
from stable_baselines3 import PPO
import os
import rl
from vehicle import Otter

RL = False

if RL == True:
    """
    RL parameters
    """
    dt = 0.02   # Time step size

    models_dir = "models/"
    model_type = "PPO"

    assert os.path.exists(
        models_dir + model_type), f"No model directory found at {models_dir + model_type}"

    vehicle = Otter(dt=dt)
    env = rl.env.SimpleEnv()

    model_path = f"{models_dir}/{model_type}/32000"
    model = PPO.load(model_path, env=env)

else:
    """
    Standard simulation parameters
    """
