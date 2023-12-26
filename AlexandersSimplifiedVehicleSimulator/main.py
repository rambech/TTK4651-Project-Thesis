"""
Main script for running Vehicle simulator

"""
from stable_baselines3 import PPO, TD3
import os
import numpy as np
from rl.env import ForwardDockingEnv
from maps import SimpleMap, Target
from vehicle import Otter

# TODO: Make it possible to add disturbances using keystrokes,
#       with side arrows determining direction and up and down
#       determining the magnitude
# TODO: Prompt user to input a test-name
# TODO: Save test data in a file for later plotting
# TODO: Make plotting tools for later plotting

RL = False

if RL == True:
    """
    RL parameters
    """
    dt = 0.02   # Time step size

    model_type = "PPO"
    load_iteration = "100000"
    timestep_multiplier = 5
    SECONDS = 120
    VEHICLE_FPS = 60
    RL_FPS = 20
    EPISODES = 10000
    TIMESTEPS = SECONDS*RL_FPS*timestep_multiplier
    seed = 1
    eta_init = np.array([0, 0, 0, 0, 0, 0], float)
    eta_d = np.array([25-0.75-1, 0, 0, 0, 0, 0], float)

    # Initialize vehicle
    vehicle = Otter(dt=1/VEHICLE_FPS)

    map = SimpleMap()
    target = Target(eta_d, vehicle.L, vehicle.B, vehicle.scale, map.origin)
    env = ForwardDockingEnv(vehicle, map, target, seed=seed,
                    render_mode="human", FPS=RL_FPS)

    models_dir = f"models"
    model_path = f"{models_dir}/{model_type}/{load_iteration}.zip"

    assert (
        os.path.exists(model_path)
    ), f"{model_path} does not exist"

    if model_type == "PPO":
        model = PPO.load(model_path, env=env)

    elif model_type == "TD3":
        model = TD3.load(model_path, env=env)

    episodes = 10

    for ep in range(episodes):
        obs = env.reset()
        terminated = False
        while not terminated:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, terminated, trunc, info = env.step(action)

    env.close()

else:
    """
    Standard simulation parameters
    """
