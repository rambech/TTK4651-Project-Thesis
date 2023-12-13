"""
Simple simulator for driving round boats
github: @rambech

Reference frames
----------------
North-east-down (NED):
    Sometimes denoted {n} is the world reference frame of the 
    vehicle simulation

Body-fixed (BODY):
    Vehicle fixed reference frame, sometimes denoted {b}

Screen:
    The computer screen reference frame, sometimes denoted {s}
"""

import pygame
from vehicle import Otter
from maps import SimpleMap, Target
import numpy as np
from utils import attitudeEuler, B2N, N2B

# Keystroke inputs
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

# TODO: Add comments
# TODO: Add function/method descriptions


class Simulator():
    """
    Simple simulator for testing vehicle dynamics, mostly for the Otter vehicle

    Attributes
    ----------
    vehicle : Vehicle
        Simulated vehicle
    map : Map
        Simulated environment/map
    fps : float
        Simulation refresh rate
    dt : float
        Sample time
    eta_init : np.ndarray
        Initial vehicle pose in NED frame
    eta : np.ndarray
        Current vehicle pose in NED frame
    nu : np.ndarray
        Current vehicle velocity
    u : np.ndarray
        Current control forces
    clock : pygame.time.Clock()
        Simulation clock
    screen : pygame.display
        Simulation screen for visualisation
    quay : SimpleQuay
        Quay the vehicle docks to and interacts with
    target : Target
        Target/desired vehicle pose
    vessel_rect: pygame.Rect
        Vehicle hitbox for environment interaction
    bounds : list[pygame.Rect]
        Environment outer bounds

    Methods
    -------
    simulate()
        Runs the main simulation loop as a pygame instance
    step(tau_d: np.ndarray)
        Calls the vehicle step function and 
        saves the resulting eta, nu and u vectors
    render()
        Renders the vehicle to screen with updated pose
    bump()
        Simulates a fully elastic collision with the quay
    close()
        Closes the display and ends the pygame instance
    """

    def __init__(self, vehicle: Otter, map: SimpleMap, target: Target = None, eta_init=np.zeros(6, float), fps=30) -> None:
        """
        Initialises simulator object

        Parameters
        ----------
        vehicle: Otter
            Simulated vehicle
        map : Map
            Simulated environment/map
        target : Target, optional
            Target/desired vehicle pose (default is None)
        eta_init : np.ndarray, optional
            Initial vehicle pose in NED frame (default is np.zeros(6, float))
        fps : float, optional
            Simulation refresh rate (default is 30)


        Returns
        ------
            None
        """

        self.vehicle = vehicle
        self.map = map
        self.fps = fps
        self.dt = 1/self.fps

        # Initial conditions
        self.eta_init = eta_init.copy()  # Save initial pose
        self.eta = eta_init              # Initialize pose
        self.nu = np.zeros(6, float)     # Init velocity
        self.u = np.zeros(2, float)      # Init control vector

        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Otter Simulator")
        self.clock = pygame.time.Clock()

        # Make a screen and fill it with a background colour
        self.screen = pygame.display.set_mode(
            [self.map.BOX_WIDTH, self.map.BOX_LENGTH])
        self.screen.fill(self.map.OCEAN_BLUE)
        self.quay = self.map.quay

        # Add target
        self.target = target

        # Initialize hitboxes
        self.vessel_rect = self.vehicle.vessel_image.get_rect()
        self.bounds = []
        self.render()

    def simulate(self):
        """
        Runs the main simulation loop as a pygame instance
        """

        # Run until the user asks to quit or hit something they shouldn't
        running = True
        out_of_bounds = False
        while running:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    # Quit if escape key is pressed
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
                        # Note: use copy() when copying arrays
                        self.eta = self.eta_init.copy()
                        self.nu = np.zeros(6, float)
                        self.u = np.zeros(2, float)
                else:
                    X = 0   # [N]
                    N = 0   # [Nm]

                # Quit if window is closed by user
                if event.type == QUIT:
                    running = False

            if self.vehicle != None:
                # Manual force input
                tau_d = np.array([X, N])

                if self.vessel_rect.colliderect(self.quay.rect):
                    # Stop simulation if the quay is hit too hard
                    if np.linalg.norm(self.nu[0:3], 3) > 0.5:
                        running = False

                    # Otherwise bump
                    else:
                        self.bump()

                # End simulation if out of bounds
                for bound in self.bounds:
                    if self.vessel_rect.colliderect(bound):
                        out_of_bounds = True
                        running = False

                # Step vehicle simulation
                if not out_of_bounds:
                    self.step(tau_d)

            self.render()
        self.close()

    def step(self, tau_d: np.ndarray):
        """
        Calls the vehicle step function and 
        saves the resulting eta, nu and u vectors

        Parameters
        ----------
        tau_d: np.ndarray
            Vector of desired surge force X and yaw momentum N
        """

        # Kinetic step
        self.nu, self.u = self.vehicle.step(
            self.eta, self.nu, self.u, tau_d, self.map.SIDESLIP, self.map.CURRENT_MAGNITUDE)
        # Kinematic step
        self.eta = attitudeEuler(self.eta, self.nu, self.dt)

    def render(self):
        """
        Updates screen with the changes that has happened with
        the vehicle and map/environment
        """

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

    def bump(self):
        """
        Simulates a fully elastic collision between the quay and the vessel
        """

        # Transform nu from {b} to {n}
        nu_n = B2N(self.eta).dot(self.nu)

        # Send the vessel back with the same speed it came in
        U_n = np.linalg.norm(nu_n[0:3], 3)
        min_U_n = min(-U_n, U_n)

        # Necessary angles
        beta = np.arctan(nu_n[2]/nu_n[0])   # Sideslip
        alpha = np.arcsin(nu_n[1]/min_U_n)  # Angle of attack

        nu_n[0:3] = np.array([min_U_n*np.cos(alpha)*np.cos(beta),
                              min_U_n*np.sin(beta),
                              min_U_n*np.sin(alpha)*np.cos(beta)])
        self.nu = N2B(self.eta).dot(nu_n)

    def close(self):
        pygame.display.quit()
        pygame.quit()


def test_simulator():
    """
    Procedure for testing simulator
    """
    # Initialize constants
    fps = 50
    eta_init = np.array([0, 0, 0, 0, 0, 0], float)
    eta_d = np.array([10, 0, 0, 0, 0, 0], float)

    # Initialize vehicle
    vehicle = Otter(dt=1/fps)

    map = SimpleMap()
    target = Target(eta_d, vehicle.L, vehicle.B, vehicle.scale, map.origin)
    simulator = Simulator(vehicle, map, target, eta_init=eta_init, fps=fps)
    simulator.simulate()


test_simulator()
