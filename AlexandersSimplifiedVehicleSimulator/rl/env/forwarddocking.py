"""
This is the simplest python RL environment I could think of.
With a large boundary around this is simply snake with one pixel


Observations space
------------------
s = [delta_x, delta_y, delta_psi, u, v, r, d_q, psi_q, d_o, psi_o]


Action space
------------
a = [n_1, n_2]


Reward
------
Max. reward per timestep: 0.4
Min. reward per timestep: -0.4
Max. reward without end conditions: 1000
Min. reward without end conditions: -1000

"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from .env import Env
from vehicle import Otter
from maps import SimpleMap, Target
from utils import attitudeEuler, D2L, D2R, N2B, B2N, ssa, R2D

from rl.rewards import r_euclidean, r_time, r_surge, r_gaussian, r_pos_e, r_heading

# TODO: Double check psi_q
# TODO: Make rewards with known lower and upper limits
# TODO: Use max time reward

# Environment parameters
FPS = 20        # [fps] Frames per second

# I have chosen the origin of NED positon to be
# in the middle of the screen. This means that
# the pygame coordinates are different to the
# NED ones, RL and everything else is calculated
# in NED, only rendering happens in the other coordinates.


class ForwardDockingEnv(Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": FPS,
    }

    def __init__(self, vehicle: Otter, map: SimpleMap, seed: int = None, render_mode=None, FPS: int = 50, eta_init=np.zeros(6, float),  docked_threshold=[1, D2R(10)]) -> None:
        super(ForwardDockingEnv, self).__init__()
        """
        Initialises ForwardDockingEnv() object
        """
        self.vehicle = vehicle
        self.map = map
        self.eta_d = np.array([25/2-0.75-1, 0, 0, 0, 0, 0], float)
        self.target = Target(self.eta_d, vehicle.L,
                             vehicle.B, map.scale, map.origin)
        self.thres = docked_threshold
        self.fps = FPS
        self.metadata["render_fps"] = FPS
        self.dt = self.vehicle.dt
        self.bounds = self.map.bounds
        self.edges = self.map.colliding_edges
        self.closest_edge = None
        self.corners = None

        self.seed = seed

        # Add obstacles
        self.obstacles = self.map.obstacles

        # Add quay
        self.quay = self.map.quay

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initial conditions
        if seed is not None:
            # Make random initial condition and weather
            np.random.seed(seed)
            self.eta = self.random_eta()

        else:
            # Use default initial conditions and weather
            self.eta_init = eta_init.copy()  # Save initial pose
            self.eta = eta_init              # Initialize pose
            self.nu = np.zeros(6, float)     # Init velocity
            self.u = np.zeros(2, float)      # Init control vector
            self.V_c = self.map.CURRENT_MAGNITUDE
            self.beta_c = self.map.SIDESLIP

        # Action space is given through super init
        N_min, E_min, N_max, E_max = self.bounds
        self.eta_max = np.array([N_max, E_max, vehicle.limits["psi_max"]])
        self.eta_min = np.array([N_min, E_min, vehicle.limits["psi_min"]])
        self.nu_max = vehicle.limits["nu_max"]
        self.nu_min = vehicle.limits["nu_min"]

        # Maximum distance and angle to quay
        d_q_max = np.linalg.norm(np.asarray(
            self.bounds[2:]) - np.asarray(self.bounds[:2]), 2)
        psi_q_max = np.pi
        psi_q_min = -psi_q_max

        # Maximum distance to each quay corner
        d_q_c_max = d_q_max

        # Maximum distance and angle to obstacle
        # d_o_max = d_q_max
        # psi_o_max = psi_q_max
        # psi_o_min = psi_q_min

        # upper = np.concatenate(
        #     (self.eta_max, self.nu_max, d_q_max, psi_q_max, d_o_max, psi_o_max), axis=None).astype(np.float32)
        # lower = np.concatenate(
        #     (self.eta_min, self.nu_min, 0, psi_q_min, 0, psi_o_min), axis=None).astype(np.float32)

        upper = np.concatenate(
            (self.eta_max, self.nu_max, d_q_max, psi_q_max, d_q_c_max, d_q_c_max), axis=None).astype(np.float32)
        lower = np.concatenate(
            (self.eta_min, self.nu_min, 0, psi_q_min, 0, 0), axis=None).astype(np.float32)

        self.observation_size = (upper.size,)

        self.observation_space = spaces.Box(
            lower, upper, self.observation_size)

        # ------------
        # Action space
        # ------------
        self.action_space = vehicle.action_space

        # --------------
        # End conditions
        # --------------
        # Fail
        time_limit = 120  # [s]
        self.step_limit = time_limit*self.fps  # [step]
        self.step_count = 0

        # Success
        s_seconds = 1
        # Must be overwritten
        self.thres = None             # [m, rad]
        self.stay_time = self.fps*s_seconds  # [step]git
        self.stay_timer = None

        # ---------
        # Rendering
        # ---------
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            # Initialize pygame
            pygame.init()
            pygame.display.set_caption("Otter RL")
            self.clock = pygame.time.Clock()

            # Make a screen and fill it with a background colour
            self.screen = pygame.display.set_mode(
                [self.map.BOX_WIDTH, self.map.BOX_LENGTH])
            self.screen.fill(self.map.OCEAN_BLUE)

    def step(self, action):
        terminated = False
        self.step_count += 1

        beta_c, V_c = self.current_force()

        # Simulate vehicle at a higher rate than the RL step
        step_rate = 1/(self.dt*self.fps)
        assert (
            step_rate % 1 == 0
        ), f"Step rate must be a positive integer, got {step_rate}. \
            Make sure the vehicle FPS is a multiple of the RL FPS"

        # Step simulator
        for _ in range(int(step_rate)):
            self.nu, self.u = self.vehicle.rl_step(
                self.eta, self.nu, self.u, action, beta_c, V_c)
            self.eta = attitudeEuler(self.eta, self.nu, self.dt)
            self.corners = self.vehicle.corners(self.eta)
            if self.crashed():
                break

        # -----------
        # Observation
        # -----------
        observation = self.get_observation()

        # -------
        # Rewards
        # -------
        # shape = 10 * r_euclidean(observation)  # + r_surge(observation)
        reward = 0
        dist = r_euclidean(observation)

        if self.prev_dist is not None:
            reward += dist - self.prev_dist
        self.prev_dist = dist

        # reward += 0.4 * (r_pos_e(observation) +
        #                  r_heading(observation, self.eta[-1]))
        reward += r_time()

        # if self.prev_shape:
        #     reward += shape - self.prev_shape

        # self.prev_shape = shape

        if self.docked():
            if self.stay_timer is None:
                self.stay_timer = 0
            else:
                self.stay_timer += 1

            # Give reward if inside area
            print(f"Steps docked: {self.stay_timer}")
            reward += 1
        else:
            self.stay_timer = None

        # if self.step_count >= self.step_limit:
        #     terminated = True
        #     reward = -100

        if self.success():
            terminated = True
            reward = 1000

        if self.time_out():
            terminated = True
            reward = -10

        if self.crashed():
            terminated = True
            reward = -10000

        if self.render_mode == "human":
            self.render()
            for obstacle in self.obstacles:
                self.screen.blit(obstacle.surf, obstacle.rect)
            self.screen.blit(self.quay.surf, self.quay.rect)

        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def get_observation(self):
        delta_eta = self.eta - self.eta_d
        west_corner, east_corner = self.quay.colliding_edge
        delta_eta_2D = np.concatenate(
            (delta_eta[0:2], ssa(delta_eta[-1])), axis=None)
        d_q, psi_q = self.direction_and_angle_to_quay()
        d_c_w = np.linalg.norm(self.eta[0:2] - np.asarray(west_corner))
        d_c_e = np.linalg.norm(self.eta[0:2] - np.asarray(east_corner))

        return np.concatenate((delta_eta_2D, self.nu[0:3], d_q, psi_q, d_c_w, d_c_e),
                              axis=None).astype(np.float32)

    def crashed(self) -> bool:
        self.find_closest_edge()
        for corner in self.corners:
            _, dist_corner_quay = D2L(self.quay.colliding_edge, corner)
            _, dist_corner_obs = D2L(self.closest_edge, corner)
            if dist_corner_obs <= 0.01:  # If vessel touches obstacle, simulation stops
                return True
            elif abs(corner[0]) >= self.eta_max[0] or abs(corner[1]) >= self.eta_max[1]:
                return True
            elif dist_corner_quay < 0.01:
                self.bump()
            else:
                continue

        return False

    def docked(self) -> bool:
        fp_corner = np.asarray(self.corners[-1])
        fs_corner = np.asarray(self.corners[0])
        _, d_c_fp = D2L(self.quay.colliding_edge, fp_corner)
        _, d_c_fs = D2L(self.quay.colliding_edge, fs_corner)
        # print(f"d_c_fp: {d_c_fp}")
        # print(f"d_c_fs: {d_c_fs}")

        if d_c_fp <= 0.1 and d_c_fs <= 0.1:
            print("Docked!")
            return True
        else:
            return False

    def time_out(self):
        return True if self.step_count >= self.step_limit else False

    def bump(self):
        """
        Simulates a fully elastic collision between the quay and the vessel
        """

        # Transform nu from {b} to {n}
        nu_n = B2N(self.eta).dot(self.nu)

        # Send the vessel back with the same speed it came in
        U_n = np.linalg.norm(nu_n[0:3], 3)
        min_U_n = -U_n

        # Necessary angles
        beta = np.arctan(nu_n[2]/nu_n[0])   # Sideslip
        alpha = np.arcsin(nu_n[1]/min_U_n)  # Angle of attack

        nu_n[0:3] = np.array([min_U_n*np.cos(alpha)*np.cos(beta),
                              min_U_n*np.sin(beta),
                              min_U_n*np.sin(alpha)*np.cos(beta)])
        self.nu = N2B(self.eta).dot(nu_n)

    def random_eta(self):
        """
        Spawn vehicle based on uniform distribution. 
        2 meter buffer at the edges 

        Parameters
        ----------
        self

        Returns
        -------
        eta_init : np.ndarray
            Random initial position

        """
        padding = 2  # [m]
        x_init = np.random.uniform(
            self.bounds[0] + padding, self.bounds[2] - self.quay.length - padding)
        y_init = np.random.uniform(
            self.bounds[1] + padding, self.bounds[3] - padding)
        ang2d = np.arctan2(
            y_init - self.eta_d[1], x_init - self.eta_d[0]) - np.pi
        # psi_init = np.random.uniform(ang2d-np.pi/2, ang2d+np.pi/2)
        psi_init = np.random.uniform(ang2d, ang2d)

        return np.array([x_init, y_init, 0, 0, 0, psi_init], float)

    # def direction_and_angle_to_obs(self):
    #     angle = 0
    #     dist = np.inf
    #     for edge in self.edges:
    #         bearing, range = D2L(edge, self.eta[0:2])
    #         if range < dist:
    #             angle = bearing - self.eta[-1]
    #             dist = range
    #             self.closest_edge = edge

    #     return dist, angle

    def find_closest_edge(self):
        dist = np.inf
        for edge in self.edges:
            _, range = D2L(edge, self.eta[0:2])
            if range < dist:
                dist = range
                self.closest_edge = edge

    def direction_and_angle_to_quay(self):
        bearing, dist = D2L(self.quay.colliding_edge, self.eta[0:2])
        angle = bearing - self.eta[-1]

        return dist, angle
