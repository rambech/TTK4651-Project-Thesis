"""
This is the simplest python RL environment I could think of.
With a large boundary around this is simply snake with one pixel


Observations space
------------------
s = [delta_x, delta_y, delta_psi, u, v, r, d_q, psi_q, d_o, psi_o]


Action space
------------
a = [n_1, n_2]

"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

from .env import Env
from vehicle import Vehicle, Otter
from maps import SimpleMap, Target
from utils import attitudeEuler, D2L, D2R, N2B, B2N, ssa

from rl.rewards import norm

# TODO: Use SSA
# TODO: Make it possible to have the current change during an episode
# TODO: Make a docking threshold of some sort
# TODO: Implement closest obstacle measure for the observation space
# TODO: Implement distance and angle to closest point on the quay for the observation space
# TODO: Visualise actuator usage
# TODO: Optional: make observations into a dict
# TODO: Add info dict

# Environment parameters
FPS = 50                          # [fps] Frames per second

# I have chosen the origin of NED positon to be
# in the middle of the screen. This means that
# the pygame coordinates are different to the
# NED ones, RL and everything else is calculated
# in NED, only rendering happens in the other coordinates.

# Weather
# SIDESLIP = 30           # [deg]
# CURRENT_MAGNITUDE = 3   # [0]


class ForwardDockingEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": FPS,
    }

    def __init__(self, vehicle: Otter, map: SimpleMap, target: Target, seed: int = None, render_mode=None, FPS: int = 50, eta_init=np.zeros(6, float),  docked_threshold=[1, D2R(10)]) -> None:
        super(ForwardDockingEnv, self).__init__()
        """
        Initialises ForwardDockingEnv() object
        """

        self.vehicle = vehicle
        self.map = map
        self.target = target
        self.fps = FPS
        self.metadata["render_fps"] = FPS
        self.dt = self.vehicle.dt
        self.bounds = self.map.bounds
        self.edges = self.map.colliding_edges
        self.closest_edge = ((0, 0), (0, 0))
        self.eta_d = self.target.eta_d

        self.seed = seed

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

        # Maximum distance and angle to obstacle
        d_o_max = d_q_max
        psi_o_max = psi_q_max
        psi_o_min = psi_q_min

        upper = np.concatenate(
            (self.eta_max, self.nu_max, d_q_max, psi_q_max, d_o_max, psi_o_max), axis=None).astype(np.float32)
        lower = np.concatenate(
            (self.eta_min, self.nu_min, 0, psi_q_min, 0, psi_o_min), axis=None).astype(np.float32)

        self.observation_size = (upper.size,)

        self.observation_space = spaces.Box(
            lower, upper, self.observation_size)
        self.action_space = vehicle.action_space

        # Add target
        self.target = target
        self.thres = docked_threshold

        if self.render_mode == "human":
            # Initialize pygame
            pygame.init()
            pygame.display.set_caption("Otter RL")
            self.clock = pygame.time.Clock()

            # Make a screen and fill it with a background colour
            self.screen = pygame.display.set_mode(
                [self.map.BOX_WIDTH, self.map.BOX_LENGTH])
            self.screen.fill(self.map.OCEAN_BLUE)
            # self.particles = []

    def reset(self, seed=None):
        if self.seed is not None:
            self.eta = self.random_eta()
        else:
            self.eta = self.eta_init.copy()

        # self.eta = self.eta_init.copy()
        self.nu = np.zeros(6, float)
        self.u = np.zeros(2, float)

        self.has_crashed = False
        self.has_docked = False
        self.dock_timer = 0

        observation = self.get_observation()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        beta_c, V_c = self.current_force()

        # Simulate vehicle at a higher rate than the RL step
        step_rate = 1/(self.dt*self.fps)
        assert (
            step_rate % 1 == 0
        ), f"Step rate must be a positive integer, got {step_rate}. \
            Make sure the vehicle FPS is a multiple of the RL FPS"

        for _ in range(int(step_rate)):
            self.nu, self.u = self.vehicle.rl_step(
                self.eta, self.nu, self.u, action, beta_c, V_c)
            self.eta = attitudeEuler(self.eta, self.nu, self.dt)

        if self.crashed():
            terminated = True
            self.has_crashed = True
        elif self.docked():
            terminated = True
            self.has_docked = True
        else:
            terminated = False

        reward = norm(self.eta, self.eta_d, self.has_crashed, self.has_docked)
        observation = self.get_observation()

        if self.render_mode == "human":
            self.render()

        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Updates screen with the changes that has happened with
        the vehicle and map/environment
        """

        if not self.screen and self.render_mode == "human":
            # Initialize pygame
            pygame.init()
            pygame.display.set_caption("Otter RL")
            self.clock = pygame.time.Clock()

            # Make a screen and fill it with a background colour
            self.screen = pygame.display.set_mode(
                [self.map.BOX_WIDTH, self.map.BOX_LENGTH])

        self.screen.fill(self.map.OCEAN_BLUE)

        # Render obstacles
        for obstacle in self.map.obstacles:
            self.screen.blit(obstacle.surf, obstacle.rect)

        # Render target pose to screen
        self.screen.blit(self.target.image, self.target.rect)

        # Render quay to screen
        self.screen.blit(self.quay.surf, self.quay.rect)

        # Render vehicle to screen
        vessel_image, self.vessel_rect = self.vehicle.render(
            self.eta, self.map.origin)
        self.screen.blit(vessel_image, self.vessel_rect)

        # Speedometer
        U = np.linalg.norm(self.nu[0:2], 2)
        font = pygame.font.SysFont("Times New Roman", 12)
        speed = font.render(
            f"SOG: {np.round(U, 2)} [m/s]", 1, (0, 0, 0))
        self.screen.blit(speed, (10, self.map.BOX_LENGTH-20))

        # Position
        x = np.round(self.eta[0])
        y = np.round(self.eta[1])
        position = font.render(f"NED: ({x}, {y})", 1, (0, 0, 0))
        self.screen.blit(position, (10, self.map.BOX_LENGTH-32))

        # Thruster revolutions
        n1 = np.round(self.u[0])
        n2 = np.round(self.u[1])
        rpm = font.render(f"THR: ({n1}, {n2} [%])", 1, (0, 0, 0))
        self.screen.blit(rpm, (10, self.map.BOX_LENGTH-44))

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.screen:
            pygame.display.quit()
            pygame.quit()

    def get_observation(self):
        delta_eta = self.eta - self.eta_d
        delta_eta_2D = np.concatenate(
            (delta_eta[0:2], delta_eta[-1]), axis=None)
        d_q, psi_q = self.direction_and_angle_to_quay()
        d_o, psi_o = self.direction_and_angle_to_obs()

        return np.concatenate((delta_eta_2D, self.nu[0:3], d_q, psi_q, d_o, psi_o),
                              axis=None).astype(np.float32)

    def crashed(self) -> bool:
        for corner in self.vehicle.corners(self.eta):
            _, dist_corner_quay = D2L(self.quay.colliding_edge, corner)
            _, dist_corner_obs = D2L(self.closest_edge, corner)
            if dist_corner_obs < 0.01:  # If vessel touches obstacle, simulation stops
                return True
            elif abs(corner[0]) >= self.eta_max[0] or abs(corner[1]) >= self.eta_max[1]:
                return True
            elif dist_corner_quay < 0.01:
                self.bump()
            else:
                continue

        return False

    def docked(self) -> bool:
        if (np.linalg.norm(self.eta_d[0:1] - self.eta[0:1]) <= self.thres[0] and abs(self.eta_d[-1] - self.eta[-1]) <= self.thres[1]):
            return True
        else:
            return False

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

    def direction_and_angle_to_obs(self):
        angle = 0
        dist = np.inf
        for edge in self.edges:
            bearing, range = D2L(edge, self.eta[0:2])
            if range < dist:
                angle = bearing - self.eta[-1]
                dist = range
                self.closest_edge = edge

        return dist, angle

    def direction_and_angle_to_quay(self):
        bearing, dist = D2L(self.quay.colliding_edge, self.eta[0:2])
        angle = bearing - self.eta[-1]

        return dist, angle

    def current_force(self) -> tuple[float, float]:
        """
        Three modes, random changing, keystroke and random static

        Random changing
        ---------------
        The current can change in both magnitude and direction during
        an episode

        Keystroke
        ---------
        Current changes during an episode by user input via the arrow keys

        Random static
        -------------
        The current is random in magnitude and direction, but is static
        through an episode

        Parameters
        ----------
            self

        Returns
        -------
            beta_c, V_c : tuple[float, float]
                Angle and magnitude of current
        """
        return 0.0, 0.0

    def random_eta(self):
        padding = 2  # [m]
        x_init = np.random.uniform(
            self.bounds[0] + padding, self.bounds[2] - self.quay.length - padding)
        y_init = np.random.uniform(
            self.bounds[1] + padding, self.bounds[3] - padding)
        psi_init = ssa(np.random.uniform(-np.pi, np.pi))

        return np.array([x_init, y_init, 0, 0, 0, psi_init], float)
