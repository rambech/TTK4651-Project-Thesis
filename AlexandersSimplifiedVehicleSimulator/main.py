"""
Main script for running Vehicle simulator

"""
from stable_baselines3 import PPO, TD3
import os
import numpy as np
from rl.env import ForwardDockingEnv, DPEnv
from maps import SimpleMap
from vehicle import Otter

from utils import D2R

# TODO: Make it possible to add disturbances using keystrokes,
#       with side arrows determining direction and up and down
#       determining the magnitude
# TODO: Prompt user to input a test-name
# TODO: Save test data in a file for later plotting
# TODO: Make plotting tools for later plotting

# To test RL or not to test RL that is the question
RL = True

env_type = "DP"
random_weather = False
seed = 1
timestep_multiplier = 5
threshold = [1, D2R(90)]
SECONDS = 120
VEHICLE_FPS = 60
RL_FPS = 20
# EPISODES = 10000
# TIMESTEPS = SECONDS*RL_FPS  # *timestep_multiplier
eta_init = np.array([0, 0, 0, 0, 0, 0], float)
eta_d = np.array([25-0.75-1, 0, 0, 0, 0, 0], float)

# Initialize vehicle
vehicle = Otter(dt=1/VEHICLE_FPS)

map = SimpleMap()
if env_type == "docking":
    env = ForwardDockingEnv(vehicle, map, seed=seed,
                            render_mode=None, FPS=RL_FPS)

elif env_type == "DP":
    env = DPEnv(vehicle, map, seed, eta_init=eta_init, render_mode="human",
                FPS=RL_FPS, threshold=threshold, random_weather=random_weather)

if RL == True:
    """
    RL parameters
    """
    model_type = "PPO"
    folder_name = "PPO-DP-39"
    load_iteration = "9600000"

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
        cunt = 0
        while not terminated:
            action, _ = model.predict(obs)
            obs, reward, terminated, trunc, info = env.step(action)
            print(f"Timestep: {cunt}")
            print(f"Reward: {reward}")
            if env_type == "DP":
                print(f"Observation: \n \
                        delta_x:    {obs[0]} \n \
                        delta_y:    {obs[1]} \n \
                        delta_psi:  {obs[2]} \n \
                        u:          {obs[3]} \n \
                        v:          {obs[4]} \n \
                        r:          {obs[5]} \n")
            cunt += 1

    env.close()

else:
    """
    Standard simulation parameters
    """
    import pygame

    # Keystroke inputs
    from pygame.locals import (
        K_UP,
        K_DOWN,
        K_LEFT,
        K_RIGHT,
        K_ESCAPE,
        K_TAB,
        KEYDOWN,
        K_q,
        K_w,
        K_a,
        K_s,
        QUIT,
    )

    episodes = 10

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        print(f"Obs: {obs}")
        cunt = 0
        action = np.zeros(2, float)  # [-1, 1]
        while not terminated:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    # Quit if escape key is pressed
                    if event.key == K_ESCAPE:
                        terminated = True

                    # Manual forwards
                    if event.key == K_UP:
                        # Constant positive surge force
                        action = np.ones(2, float)

                    elif event.key == K_DOWN:
                        # Constant negative surge force
                        action = -np.ones(2, float)

                    elif event.key == K_RIGHT:
                        # Constant positive yaw moment
                        action = np.array([1, -1])

                    elif event.key == K_LEFT:
                        # Constant positive yaw moment
                        action = np.array([-1, 1])

                    elif event.key == K_q:
                        action = np.array([1, 0])

                    elif event.key == K_w:
                        action = np.array([0, 1])

                    elif event.key == K_a:
                        action = np.array([-1, 0])

                    elif event.key == K_s:
                        action = np.array([0, -1])

                else:
                    action = np.zeros(2, float)  # [-1, 1]

            obs, reward, terminated, trunc, info = env.step(action)
            print(f"Timestep: {cunt}")
            print(f"Reward: {reward}")
            if env_type == "DP":
                print(f"Observation: \n \
                        delta_x:    {obs[0]} \n \
                        delta_y:    {obs[1]} \n \
                        delta_psi:  {obs[2]} \n \
                        u:          {obs[3]} \n \
                        v:          {obs[4]} \n \
                        r:          {obs[5]} \n")
            cunt += 1

    env.close()
