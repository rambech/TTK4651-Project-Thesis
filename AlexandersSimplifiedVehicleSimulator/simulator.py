"""
Simple simulator for driving round boats
"""

import pygame
from vehicle import Otter
from maps import SimpleMap
import numpy as np
from utils import attitudeEuler

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    K_TAB,
    KEYDOWN,
    QUIT,
)


class Simulator():
    def __init__(self, vehicle: Otter, map: SimpleMap, eta_init=np.zeros(6, float), fps=30) -> None:
        self.vehicle = vehicle
        self.map = map
        self.fps = fps
        self.dt = 1/self.fps

        # Initial conditions
        self.eta_i = eta_init  # Save initial pose
        self.eta = eta_init    # Initialize pose
        self.nu = np.zeros(6, float)
        self.u = np.zeros(2, float)

        self.screen = None

    def simulate(self):
        # Initialize pygame
        pygame.init()

        # Make a screen and fill it with a background colour
        self.screen = pygame.display.set_mode(
            [self.map.BOX_WIDTH, self.map.BOX_LENGTH])
        self.screen.fill(self.map.OCEAN_BLUE)

        # Run until the user asks to quit
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False

                    # Manual surge motion
                    if event.key == K_UP:
                        # Constant positive surge velocity
                        u_d = 4             # [kts]
                        u_d = u_d * 0.514   # [m/s]

                    elif event.key == K_DOWN:
                        # Constant negative surge velocity
                        u_d = -4            # [kts]
                        u_d = u_d * 0.514   # [m/s]
                    else:
                        u_d = 0

                    # Manual yaw-rate
                    if event.key == K_RIGHT:
                        # Constant positive yaw-rate
                        r_d = 5   # [deg/s]

                    elif event.key == K_LEFT:
                        # Constant negative yaw-rate
                        r_d = -5
                    else:
                        r_d = 0

                    # Go back to start
                    if event.key == K_TAB:
                        # Go back to initial condition
                        pass
                else:
                    u_d = 0
                    r_d = 0

                if event.type == QUIT:
                    running = False

            if self.vehicle != None:
                nu_d = np.array([u_d, 0, r_d])
                # print(f"nu_d: {nu_d}")
                self.step(nu_d)

            self.render()

        self.close()

    def step(self, nu_d: np.ndarray):
        self.nu, self.u = self.vehicle.step(
            self.eta, self.nu, self.u, nu_d, self.map.SIDESLIP, self.map.CURRENT_MAGNITUDE)
        self.eta = attitudeEuler(self.eta, self.nu, self.dt)
        print(f"Nu:  {self.nu}")
        print(f"Eta: {self.eta}")
        print(f"u:   {self.u}")

    def render(self):
        # self.screen.fill(self.map.OCEAN_BLUE)
        for obstacle in self.map.obstacles:
            self.screen.blit(obstacle.surf, obstacle.rect)

        if self.vehicle != None:
            vessel_image, vessel_shape = self.vehicle.render(
                self.eta, (0, 0))
            self.screen.blit(vessel_image, vessel_shape)

        pygame.display.flip()

    def close(self):
        pygame.display.quit()
        pygame.quit()


def test_simulator():
    fps = 30
    vehicle = Otter(dt=1/fps)

    map = SimpleMap()
    simulator = Simulator(vehicle, map, fps=fps)
    simulator.simulate()


test_simulator()
