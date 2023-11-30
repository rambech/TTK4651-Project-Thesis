from pygame import Surface


class Wall():
    def __init__(self, wall_width, wall_length, wall_pos: tuple[float, float]) -> None:
        self.surf = Surface((wall_width, wall_length))
        self.surf.fill((0, 0, 0))   # Black
        self.rect = self.surf.get_rect(center=(wall_pos[0], wall_pos[1]))


class Map():
    # Map parameters
    BOX_WIDTH = 500     # [m]   Overall box width
    BOX_LENGTH = 500    # [m]   Overall box length/height
    WALL_THICKNESS = 2  # [m]   Border wall thickness

    def __init__(self) -> None:
        # Placements of border wall
        north_wall_pos = (self.BOX_WIDTH/2, 0 + (self.WALL_THICKNESS/2))
        south_wall_pos = (self.BOX_WIDTH/2, self.BOX_LENGTH -
                          (self.WALL_THICKNESS/2))
        east_wall_pos = (self.BOX_WIDTH -
                         (self.WALL_THICKNESS/2), self.BOX_LENGTH/2)
        west_wall_pos = (0 + (self.WALL_THICKNESS/2), self.BOX_LENGTH/2)

        # Instantiation of walls
        north_wall = Wall(
            self.BOX_WIDTH, self.WALL_THICKNESS, north_wall_pos)
        south_wall = Wall(
            self.BOX_WIDTH, self.WALL_THICKNESS, south_wall_pos)
        east_wall = Wall(self.WALL_THICKNESS,
                         self.BOX_LENGTH, east_wall_pos)
        west_wall = Wall(self.WALL_THICKNESS,
                         self.BOX_LENGTH, west_wall_pos)

        # Make obstacle list with the walls
        self.obstacles = [north_wall, south_wall, east_wall, west_wall]
