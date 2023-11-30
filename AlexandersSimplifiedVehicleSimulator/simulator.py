"""
Simple simulator for driving round boats
"""

import pygame
from vehicle import Otter
from maps import SimpleMap


class Simulator():
    def __init__(self, vehicle: Otter, map: SimpleMap) -> None:
        self.vehicle = vehicle
        self.map = map

    def simulate(self):
        # Initialize pygame
        pygame.init()

        map = self.map

        # Make a screen and fill it with a background colour
        screen = pygame.display.set_mode([map.BOX_WIDTH, map.BOX_LENGTH])
        screen.fill(map.OCEAN_BLUE)

        # Run until the user asks to quit
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for obstacle in map.obstacles:
                screen.blit(obstacle.surf, obstacle.rect)

            if self.vehicle != None:
                pass

            pygame.display.flip()

        pygame.quit()


def test_simulator():
    map = SimpleMap()
    simulator = Simulator(None, map)
    simulator.simulate()
