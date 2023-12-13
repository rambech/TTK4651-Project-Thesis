"""
This is the simplest python RL environment I could think of.
With a large boundary around this is simply snake with one pixel
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

from .env import Env
from vehicle import Vehicle, Otter
from maps import SimpleMap, Target
from utils import attitudeEuler
from rl.rewards import norm

# TODO: Use SSA
# TODO: Make it possible to have the current change during an episode
# TODO: Make a docking threshold that is (0, 1]

# Environment parameters
# BOX_WIDTH = 500                 # [m]   Overall box width
# BOX_LENGTH = 500                # [m]   Overall box length
# dt = 0.02                       # [s]   Sample time
FPS = 50                          # [fps] Frames per second
# QUAY_WIDTH = 100                # [m]
# QUAY_LENGTH = 10                # [m]
# QUAY_X_POS = BOX_WIDTH/2        # [m]   x position of the center of quay
# QUAY_Y_POS = 0 + QUAY_LENGTH/2  # [m]   given in screen coordinates
# OCEAN_BLUE = ((0, 157, 196))
# ORIGO = (BOX_WIDTH/2, BOX_LENGTH/2)

# I have chosen the origin of NED positon to be
# in the middle of the screen. This means that
# the pygame coordinates are different to the
# NED ones, RL and everything else is calculated
# in NED, only rendering happens in the other coordinates.

# Observation space parameters
N_max = 250         # [m]   Position x in NED
N_min = -250        # [m]
E_max = 250         # [m]   Position y in NED
E_min = -250        # [m]

# Weather
SIDESLIP = 30           # [deg]
CURRENT_MAGNITUDE = 3   # [0]


class SimpleEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": FPS,
    }

    def __init__(self, vehicle: Otter, map: SimpleMap, target: Target, seed: int = None, eta_init=np.zeros(6, float),  docked_threshold=[1, 10]) -> None:
        super(SimpleEnv, self).__init__()
        """
        Initialises SimpleEnv() object
        """

        self.vehicle = vehicle
        self.map = map
        self.fps = FPS
        self.dt = 1/FPS

        # Initial conditions
        self.eta_init = eta_init.copy()  # Save initial pose
        self.eta = eta_init              # Initialize pose
        self.nu = np.zeros(6, float)     # Init velocity
        self.u = np.zeros(2, float)      # Init control vector

        # Action space is given through super init
        self.eta_max = np.array([N_max, E_max, vehicle.limits["psi_max"]])
        self.eta_min = np.array([N_min, E_min, vehicle.limits["psi_min"]])
        self.nu_max = vehicle.limits["nu_max"]
        self.nu_min = vehicle.limits["nu_min"]
        # List of maximum actuator output
        u_max = vehicle.limits["u_max"]
        # List of minimum actuator output
        u_min = vehicle.limits["u_min"]

        upper = np.concatenate(
            (self.eta_max, self.nu_max, u_max), axis=None).astype(np.float32)
        lower = np.concatenate(
            (self.eta_min, self.nu_min, u_min), axis=None).astype(np.float32)

        self.observation_space = spaces.Box(lower, upper, (upper.size,))
        self.action_space = vehicle.action_space

        # Add target
        self.target = target
        self.thres = docked_threshold

        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Otter RL")
        self.clock = pygame.time.Clock()

        # Make a screen and fill it with a background colour
        self.screen = pygame.display.set_mode(
            [self.map.BOX_WIDTH, self.map.BOX_LENGTH])
        self.screen.fill(self.map.OCEAN_BLUE)
        self.quay = self.map.quay

        # For render
        self.isopen = True
        self.moving_obstacles: list = None
        # self.lander: Optional[Box2D.b2Body] = None # Is swapped out for self.vehicle
        self.particles = []

    def reset(self, seed=None):
        self.nu = self.vehicle.nu
        self.eta = self.vehicle.eta
        self.u = self.vehicle.u

        self.dt = self.vehicle.dt
        self.has_crashed = False
        self.has_docked = False

        observation = np.zeros((12, 1)).astype(np.float32)
        info = {}

        return observation, info

    def step(self, action):
        beta_c, V_c = self.current()

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

        reward = norm(self.eta, self.nu, self.has_crashed, self.has_docked)
        observation = [self.eta.tolist +
                       self.nu.tolist + self.u.tolist] + list()

        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Updates screen with the changes that has happened with
        the vehicle and map/environment
        """

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.map.BOX_WIDTH, self.map.BOX_LENGTH)
            )

        self.screen.fill(self.map.OCEAN_BLUE)
        bounds = []

        # Add outer bounds of map
        for obstacle in self.map.obstacles:
            bounds.append(obstacle.rect)
            self.screen.blit(obstacle.surf, obstacle.rect)

        # List of bounds
        self.bounds = bounds

        # Render target pose to screen
        if self.target != None:
            self.screen.blit(self.target.image, self.target.rect)

        # Render quay to screen
        self.screen.blit(self.quay.surf, self.quay.rect)

        # Render vehicle to screen
        if self.vehicle != None:
            vessel_image, self.vessel_rect = self.vehicle.render(
                self.eta, self.map.origin)
            self.screen.blit(vessel_image, self.vessel_rect)

            # Speedometer
            U = np.linalg.norm(self.nu[0:2], 2)
            font = pygame.font.SysFont("Times New Roman", 12)
            speed = font.render(f"SOG: {np.round(U, 2)} [m/s]", 1, (0, 0, 0))
            self.screen.blit(speed, (10, self.map.BOX_LENGTH-20))

            # Position
            x = np.round(self.eta[0])
            y = np.round(self.eta[1])
            position = font.render(f"NED: ({x}, {y})", 1, (0, 0, 0))
            self.screen.blit(position, (10, self.map.BOX_LENGTH-32))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.display.quit()
        pygame.quit()

        self.isopen = False

    def crashed(self) -> bool:
        if (abs(self.eta[0]) >= self.eta_max[0] or abs(self.eta[1]) >= self.eta_max[1]):
            return True
        else:
            return False

    def docked(self) -> bool:
        if (np.linalg.norm(self.eta_d[0:1] - self.eta[0:1]) <= self.thres[0] and abs(self.eta_d[2] - self.eta[2]) <= self.thres[1]):
            return True
        else:
            return False

    def current(self) -> tuple[float, float]:
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
        pass
