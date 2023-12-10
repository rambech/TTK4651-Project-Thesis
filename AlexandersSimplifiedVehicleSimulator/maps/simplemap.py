import pygame
import numpy as np
from .map import Map, Wall

# TODO: Make an outline of the goal position
#       that is marked on the map


class SimpleQuay():
    def __init__(self, quay_width, quay_length, quay_pos: tuple[float, float]):
        self.surf = pygame.Surface((quay_width, quay_length))
        self.surf.fill((192, 192, 192))
        self.rect = self.surf.get_rect(center=(quay_pos[0], quay_pos[1]))
        self.bump_idx = 0   # Push vehicle southward


class SimpleMap(Map):
    # Map parameters
    BOX_WIDTH = 700                     # [px]   Overall box width
    BOX_LENGTH = 700                    # [px]   Overall box length
    SCALE = 30                          # [px/m] pixels/meter
    QUAY_SIZE_M = (10, 0.75)            # [m]
    QUAY_SIZE = (QUAY_SIZE_M[0]*SCALE, QUAY_SIZE_M[1]*SCALE)
    # [m]   x position of the center of quay
    QUAY_X_POS = BOX_WIDTH/2
    QUAY_Y_POS = 0 + QUAY_SIZE[1]/2     # [m]   given in screen coordinates
    OCEAN_BLUE = (0, 157, 196)
    BACKGROUND_COLOR = OCEAN_BLUE
    origin = np.array([BOX_WIDTH/2, BOX_LENGTH/2, 0], float)

    # Map obstacles
    quay = SimpleQuay(QUAY_SIZE[0], QUAY_SIZE[1], (QUAY_X_POS, QUAY_Y_POS))
    extra_wall_width = (BOX_WIDTH/2)-(QUAY_SIZE[0]/2)
    extra_wall_east = Wall(extra_wall_width, QUAY_SIZE[1],
                           (BOX_WIDTH-(extra_wall_width/2), QUAY_Y_POS))
    extra_wall_west = Wall(extra_wall_width, QUAY_SIZE[1],
                           (extra_wall_width/2, QUAY_Y_POS))

    # Weather
    SIDESLIP = 0  # 30           # [deg]
    CURRENT_MAGNITUDE = 0  # 3   # [0]

    def __init__(self) -> None:
        super(SimpleMap, self).__init__()
        self.obstacles.append(self.extra_wall_east)
        self.obstacles.append(self.extra_wall_west)


def test_map():
    """
    Function for testing the map above
    """
    pygame.init()

    mymap = SimpleMap()

    screen = pygame.display.set_mode([mymap.BOX_WIDTH, mymap.BOX_LENGTH])
    screen.fill(mymap.OCEAN_BLUE)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for obstacle in mymap.obstacles:
            screen.blit(obstacle.surf, obstacle.rect)

        pygame.display.flip()

    pygame.quit()


# test_map()
