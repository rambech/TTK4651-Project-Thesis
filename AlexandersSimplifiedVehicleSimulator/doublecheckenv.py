from rl.env import SimpleEnv
from vehicle import Otter
import numpy as np
from maps import SimpleMap, Target

# Initialize constants
rl_fps = 20
vehicle_fps = 60
eta_init = np.array([0, 0, 0, 0, 0, 0], float)
eta_d = np.array([15-0.75-1, 0, 0, 0, 0, 0], float)

# Initialize vehicle
vehicle = Otter(dt=1/vehicle_fps)

map = SimpleMap()
target = Target(eta_d, vehicle.L, vehicle.B, vehicle.scale, map.origin)
env = SimpleEnv(vehicle, map, target, render_mode="human", FPS=rl_fps)

episodes = 50

for episode in range(episodes):
    terminated = False
    obs = env.reset()
    while not terminated:
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, terminated, truncated, info = env.step(random_action)
        print('reward', reward)
