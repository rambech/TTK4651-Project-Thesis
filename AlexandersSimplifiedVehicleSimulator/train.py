import os
import time
import numpy as np
from rl.env import SimpleEnv
from vehicle import Otter
from maps import SimpleMap, Target

# Training settings
model_type = "PPO"
SECONDS = 60
FPS = 50
TIMESTEPS = SECONDS*FPS
EPISODES = 10000

eta_init = np.array([0, 0, 0, 0, 0, 0], float)
eta_d = np.array([15-0.75-1, 0, 0, 0, 0, 0], float)

# Initialize vehicle
vehicle = Otter(dt=1/FPS)

map = SimpleMap()
target = Target(eta_d, vehicle.L, vehicle.B, vehicle.scale, map.origin)

if model_type == "PPO":
    from stable_baselines3 import PPO

    models_dir = f"models/PPO"
    logdir = f"logs/PPO"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = SimpleEnv(vehicle, map, target, render_mode="human", )
    env.reset()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    for i in range(1, EPISODES):
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=True, tb_log_name="PPO")
        if i % 10 == 0:
            model.save(f"{models_dir}/{TIMESTEPS*i}")
