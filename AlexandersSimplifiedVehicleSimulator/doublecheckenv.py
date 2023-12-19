from rl.env import SimpleEnv
from vehicle import Otter
import numpy as np
from maps import SimpleMap, Target
from utils import R2D

# Initialize constants
rl_fps = 20
vehicle_fps = 60
eta_init = np.array([0, 0, 0, 0, 0, 0], float)

# Initialize vehicle
vehicle = Otter(dt=1/vehicle_fps)

map = SimpleMap()
x_d = map.QUAY_POS[0] - map.QUAY_SIZE[0]/2 - vehicle.L/2
eta_d = np.array([x_d, 0, 0, 0, 0, 0], float)
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
        print(f"Observation: \n \
                delta_x:    {obs[0]} \n \
                delta_y:    {obs[1]} \n \
                delta_psi:  {obs[2]} \n \
                u:          {obs[3]} \n \
                v:          {obs[4]} \n \
                r:          {obs[5]} \n \
                d_q:        {obs[6]} \n \
                psi_q:      {R2D(obs[7])} \n \
                d_o:        {obs[8]} \n \
                psi_o:      {R2D(obs[9])} \n")
