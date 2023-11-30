import pygame
from .map import Map

# TODO: Make a map base class
# TODO: Program a simple quay map
# TODO: Convert stuff to NED?
# TODO: Make an outline of the goal position
#       that is marked on the map


class SimpleQuay():
    def __init__(self, quay_width, quay_length, quay_pos: tuple[float, float]):
        self.surf = pygame.Surface((quay_width, quay_length))
        self.surf.fill((192, 192, 192))
        self.rect = self.surf.get_rect(center=(quay_pos[0], quay_pos[1]))


class SimpleMap(Map):
    # Map parameters
    BOX_WIDTH = 500                 # [m]   Overall box width
    BOX_LENGTH = 500                # [m]   Overall box length
    QUAY_WIDTH = 100                # [m]
    QUAY_LENGTH = 10                # [m]
    QUAY_X_POS = BOX_WIDTH/2        # [m]   x position of the center of quay
    QUAY_Y_POS = 0 + QUAY_LENGTH/2  # [m]   given in screen coordinates
    OCEAN_BLUE = (0, 157, 196)
    BACKGROUND_COLOR = OCEAN_BLUE
    ORIGO = (BOX_WIDTH/2, BOX_LENGTH/2)

    # Map obstacles
    quay = SimpleQuay(QUAY_WIDTH, QUAY_LENGTH, (QUAY_X_POS, QUAY_Y_POS))

    # Weather
    SIDESLIP = 30           # [deg]
    CURRENT_MAGNITUDE = 3   # [0]

    def __init__(self) -> None:
        super(SimpleMap, self).__init__()
        self.obstacles.append(self.quay)


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


test_map()
