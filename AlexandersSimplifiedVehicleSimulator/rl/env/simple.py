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
from utils import attitudeEuler
from rl.rewards import norm

# TODO Use SSA
# TODO Allow the boat to touch the quay if the speed is low enough

# Environment parameters
BOX_WIDTH = 500                 # [m]   Overall box width
BOX_LENGTH = 500                # [m]   Overall box length
dt = 0.1                        # [s]   Sample time
FPS = 1/dt                      # [fps] Frames per second
QUAY_WIDTH = 100                # [m]
QUAY_LENGTH = 10                # [m]
QUAY_X_POS = BOX_WIDTH/2        # [m]   x position of the center of quay
QUAY_Y_POS = 0 + QUAY_LENGTH/2  # [m]   given in screen coordinates
OCEAN_BLUE = ((0, 157, 196))
ORIGO = (BOX_WIDTH/2, BOX_LENGTH/2)

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


class SimpleQuay():
    def __init__(self):
        self.surf = pygame.Surface((QUAY_WIDTH, QUAY_LENGTH))
        self.surf.fill((192, 192, 192))
        self.rect = self.surf.get_rect(center=(QUAY_X_POS, QUAY_Y_POS))


class SimpleEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": FPS,
    }

    def __init__(self, vehicle: Otter, seed=None, eta_d=np.zeros((3, 1), float), docked_threshold=[1, 10]) -> None:
        super(SimpleEnv, self).__init__()
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

        self.thres = docked_threshold
        self.vehicle = vehicle
        self.eta_d = eta_d

        self.init_eta = np.zeros((3, 1))

        # For render
        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.quay = SimpleQuay()
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

        pygame.init()
        pygame.display.init()
        self.screen = pygame.Surface((BOX_WIDTH, BOX_LENGTH))
        self.screen.fill((0, 157, 196))

        observation = np.zeros((12, 1)).astype(np.float32)
        info = {}

        return observation, info

    def step(self, action):
        beta_c = SIDESLIP           # TODO: Replace with function
        V_c = CURRENT_MAGNITUDE     # TODO: Replace with function

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
        self.screen.fill((OCEAN_BLUE))
        vessel_image, vessel_shape = self.vehicle.render_update(self.eta)
        self.screen.blit(vessel_image + ORIGO[0], vessel_shape + ORIGO[1])
        self.screen.blit(self.quay.surf + ORIGO[0], self.quay.rect + ORIGO[1])

        pygame.display.flip()

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
