import os
import time
import numpy as np
from rl.env import SimpleEnv
from vehicle import Otter
from maps import SimpleMap, Target

# Training settings
model_type = "PPO"
timestep_multiplier = 5
SECONDS = 120
VEHICLE_FPS = 60
RL_FPS = 20
EPISODES = 10000
TIMESTEPS = SECONDS*RL_FPS*timestep_multiplier
print(f"Timesteps: {TIMESTEPS}")

seed = 1
eta_init = np.array([0, 0, 0, 0, 0, 0], float)
eta_d = np.array([15-0.75-1, 0, 0, 0, 0, 0], float)

# Initialize vehicle
vehicle = Otter(dt=1/VEHICLE_FPS)

map = SimpleMap()
target = Target(eta_d, vehicle.L, vehicle.B, vehicle.scale, map.origin)
env = SimpleEnv(vehicle, map, target, seed=seed, render_mode=None, FPS=RL_FPS)

if model_type == "PPO":
    from stable_baselines3 import PPO

    models_dir = f"models/PPO"
    logdir = f"logs/PPO"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env.reset(seed)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    for episode in range(1, EPISODES):
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False, tb_log_name="PPO")
        if episode % 10 == 0:
            model.save(f"{models_dir}/{TIMESTEPS*episode}")

    # for episode in range(1, EPISODES):
    #     env.reset()
    #     terminated = False
    #     while not terminated:
    #         obs, reward, terminated, trunc, info = env.step(
    #             env.action_space.sample())

    #     if episode % 10 == 0:
    #         model.save(f"{models_dir}/{TIMESTEPS*episode}")
