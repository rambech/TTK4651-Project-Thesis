"""
Simple simulator for driving round boats
"""

import pygame
from vehicle import Otter


class Simulator():
    def __init__(self, vehicle: Otter, map) -> None:
        pass

    def simulate(self):
        # Initialize pygame
        pygame.init()

        # Make a screen
        screen = pygame.display.set_mode([500, 500])

        # Run until the user asks to quit
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.quit:
                    running = False

        pygame.quit()
