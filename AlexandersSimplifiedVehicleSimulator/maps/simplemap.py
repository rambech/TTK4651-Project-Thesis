from pygame import Surface

# TODO: Make a map base class
# TODO: Program a simple quay map


class SimpleMap():
    # Map parameters
    BOX_WIDTH = 500                 # [m]   Overall box width
    BOX_LENGTH = 500                # [m]   Overall box length
    QUAY_WIDTH = 100                # [m]
    QUAY_LENGTH = 10                # [m]
    QUAY_X_POS = BOX_WIDTH/2        # [m]   x position of the center of quay
    QUAY_Y_POS = 0 + QUAY_LENGTH/2  # [m]   given in screen coordinates
    OCEAN_BLUE = ((0, 157, 196))
    ORIGO = (BOX_WIDTH/2, BOX_LENGTH/2)

    def __init__(self) -> None:
        pass
