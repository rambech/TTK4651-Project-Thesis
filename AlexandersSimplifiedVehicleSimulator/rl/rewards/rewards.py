import numpy as np


def norm(eta, eta_d, has_crashed, has_docked):
    if has_crashed:
        reward = -10000
    elif has_docked:
        reward = 10000
    else:
        r1 = -np.linalg.norm(eta_d[0:2] - eta[0:2], 2)
        r2 = -abs(eta[-1] - eta_d[-1])
        reward = 10*(r1 + r2)

    return reward


def r_psi_e(psi_e):
    """
    r = Ce^{-1/(2*sigma^2) pos_e^2}

    Parameters
    ----------
    psi_e : float
        Heading error

    Returns
    -------
    r : float
        Gaussian heading reward
    """
    sigma = np.pi/4  # [rad]
    C = 10           # Max. along axis reward

    return C*np.exp(-1/(2*sigma**2) * psi_e**2)


def r_pos_e(pos_e):
    """
    r = Ce^{-1/(2*sigma^2) pos_e^2}

    """
    sigma = 5   # [m]
    C = 20      # Max. along axis reward

    return C*np.exp(-1/(2*sigma**2) * pos_e**2)


def r_time():
    return -10


def gaussian(obs):
    """
    r = C1e^{-1/(2*sigma^2) x_e^2} + C1e^{-1/(2*sigma^2) y_e^2} + C2e^{-1/((2*sigma^2) y_e^2}

    """
    return r_pos_e(obs[0]) + r_pos_e(obs[1]) + r_psi_e(obs[2])
