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

        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Otter Simulator")
        self.clock = pygame.time.Clock()

        # Make a screen and fill it with a background colour
        self.screen = pygame.display.set_mode(
            [self.map.BOX_WIDTH, self.map.BOX_LENGTH])
        self.screen.fill(self.map.OCEAN_BLUE)

    def simulate(self):
        # Run until the user asks to quit
        running = True
        out_of_bounds = False
        while running:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False

                    # Manual surge force
                    if event.key == K_UP:
                        # Constant positive surge force
                        X = 5   # [N]

                    elif event.key == K_DOWN:
                        # Constant negative surge force
                        X = -5  # [N]

                    else:
                        X = 0   # [N]

                    # Manual yaw moment
                    if event.key == K_RIGHT:
                        # Constant positive yaw moment
                        N = 5   # [Nm]

                    elif event.key == K_LEFT:
                        # Constant positive yaw moment
                        N = -5  # [Nm]

                    else:
                        N = 0   # [Nm]

                    # Go back to start
                    if event.key == K_TAB:
                        # Go back to initial condition
                        pass
                else:
                    X = 0
                    N = 0

                if event.type == QUIT:
                    running = False

            if self.vehicle != None:
                tau_d = np.array([X, N])
                self.step(tau_d)

            self.render()

        self.close()

    def step(self, nu_d: np.ndarray):
        self.nu, self.u = self.vehicle.step(
            self.eta, self.nu, self.u, nu_d, self.map.SIDESLIP, self.map.CURRENT_MAGNITUDE)
        self.eta = attitudeEuler(self.eta, self.nu, self.dt)
        # print(f"Nu:  {self.nu}")
        # print(f"Eta: {self.eta}")
        # print(f"u:   {self.u}")

    def render(self):
        self.screen.fill(self.map.OCEAN_BLUE)
        for obstacle in self.map.obstacles:
            self.screen.blit(obstacle.surf, obstacle.rect)

        if self.vehicle != None:
            vessel_image, image_pos = self.vehicle.render(
                self.eta, self.map.SCALE, self.map.ORIGO)
            self.screen.blit(vessel_image, image_pos)

        pygame.display.flip()
        self.clock.tick(self.fps)

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
