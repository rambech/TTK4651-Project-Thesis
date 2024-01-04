import os
import time
import json
import numpy as np
from rl.env import ForwardDockingEnv, DPEnv
from vehicle import Otter
from maps import SimpleMap, Target
from utils import D2R
from stable_baselines3.common.callbacks import CheckpointCallback

# TODO: Make every model save to a new numbered folder,
#       like the logs do automatically
# TODO: Add a settings file and put it into model folder

# Training settings
model_type = "PPO"
env_type = "docking"
random_weather = False
seed = 1
threshold = 3
timestep_multiplier = 5
SECONDS = 120
VEHICLE_FPS = 60
RL_FPS = 20
EPISODES = 10000
TIMESTEPS = SECONDS*RL_FPS*EPISODES  # *timestep_multiplier
print(f"Timesteps: {TIMESTEPS}")

test_name = input(f"Test name is {model_type}-{env_type}-")
test_name = f"{model_type}-{env_type}-{test_name}"

models_dir = "models"
log_dir = "logs"
model_path = f"{models_dir}/{test_name}"
log_path = f"{log_dir}/{test_name}"

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(log_path):
    os.makedirs(log_path)

file_name = f"{test_name}.json"
file_path = os.path.join(model_path, file_name)


# Initialize vehicle
vehicle = Otter(dt=1/VEHICLE_FPS)

map = SimpleMap()
# target = Target(eta_d, vehicle.L, vehicle.B, vehicle.scale, map.origin)

# User input
if env_type == "docking":
    eta_init = np.array([0, 0, 0, 0, 0, 0], float)
    env = ForwardDockingEnv(vehicle, map, seed=seed,
                            render_mode=None, FPS=RL_FPS)

elif env_type == "DP":
    eta_init = np.array([0, 0, 0, 0, 0, 0], float)
    env = DPEnv(vehicle, map, seed, render_mode=None,
                FPS=RL_FPS, threshold=threshold, random_weather=random_weather)

data = {
    "Test_name": test_name,
    "Model type": model_type,
    "Env type": env_type,
    "Random Weather": random_weather,
    "Vehicle fps": VEHICLE_FPS,
    "RL fps": RL_FPS,
    "Episodes": EPISODES,
    "Timesteps": TIMESTEPS,
    "Threshold": threshold,
    "Seed": seed,
    "Initial pose": eta_init.tolist(),
    "Commit hash": "1ac5e92"
}

# Save the dictionary to the file
with open(file_path, 'w') as json_file:
    json.dump(data, json_file, indent=2)

checkpoint_callback = CheckpointCallback(
    save_freq=RL_FPS*SECONDS*10,    # Save every tenth episode
    save_path=model_path,
    name_prefix=test_name,
    save_replay_buffer=True,
    save_vecnormalize=True,
)

env.reset(seed)
if model_type == "PPO":
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

if model_type == "TD3":
    from stable_baselines3 import TD3

    model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=log_path)


model.learn(total_timesteps=TIMESTEPS,
            callback=checkpoint_callback, tb_log_name=model_type)

# for episode in range(1, EPISODES+1):
#     model.learn(total_timesteps=TIMESTEPS,
#                 reset_num_timesteps=False, tb_log_name=model_type)
#     if episode % 10 == 0:
#         model.save(f"{model_path}/{TIMESTEPS*episode}")


# for episode in range(1, EPISODES):
#     env.reset()
#     terminated = False
#     while not terminated:
#         obs, reward, terminated, trunc, info = env.step(
#             env.action_space.sample())

#     if episode % 10 == 0:
#         model.save(f"{models_dir}/{TIMESTEPS*episode}")
