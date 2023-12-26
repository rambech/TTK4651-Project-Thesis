"""
Main script for running Vehicle simulator

"""
from stable_baselines3 import PPO, TD3
import os
import numpy as np
from rl.env import ForwardDockingEnv, DPEnv
from maps import SimpleMap
from vehicle import Otter

# TODO: Make it possible to add disturbances using keystrokes,
#       with side arrows determining direction and up and down
#       determining the magnitude
# TODO: Prompt user to input a test-name
# TODO: Save test data in a file for later plotting
# TODO: Make plotting tools for later plotting

RL = True

if RL == True:
    """
    RL parameters
    """
    dt = 0.02   # Time step size

    model_type = "PPO"
    env_type = "DP"
    folder_name = "PPO-DP-1"
    load_iteration = "10320000"
    timestep_multiplier = 5
    SECONDS = 120
    VEHICLE_FPS = 60
    RL_FPS = 20
    EPISODES = 10000
    TIMESTEPS = SECONDS*RL_FPS*timestep_multiplier
    seed = 5
    eta_init = np.array([0, 0, 0, 0, 0, 0], float)
    eta_d = np.array([25-0.75-1, 0, 0, 0, 0, 0], float)

    # Initialize vehicle
    vehicle = Otter(dt=1/VEHICLE_FPS)

    map = SimpleMap()
    if env_type == "docking":
        env = ForwardDockingEnv(vehicle, map, seed=seed,
                                render_mode=None, FPS=RL_FPS)

    elif env_type == "DP":
        env = DPEnv(vehicle, map, seed, render_mode="human", FPS=RL_FPS)

    models_dir = f"models"
    model_path = f"{models_dir}/{folder_name}/{load_iteration}.zip"
    assert (
        os.path.exists(model_path)
    ), f"{model_path} does not exist"

    if model_type == "PPO":
        model = PPO.load(model_path, env=env)

    elif model_type == "TD3":
        model = TD3.load(model_path, env=env)

    episodes = 10

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        print(f"Obs: {obs}")
        while not terminated:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, terminated, trunc, info = env.step(action)
            print(f"Reward: {reward}")
    env.close()

else:
    """
    Standard simulation parameters
    """
