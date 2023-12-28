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


def r_psi_e(psi_e, pos_e):
    """
    r = Ce^{-1/(2*sigma^2) pos_e^2}

    Parameters
    ----------
    psi_e : float
        Heading error
    pos_e : np.ndarray
        Position error in x and y

    Returns
    -------
    r : float
        Gaussian heading reward
    """

    if np.linalg.norm(pos_e, 2) <= 3:
        sigma = np.pi/4  # [rad]
        C = 1            # Max. along axis reward

        return C*np.exp(-1/(2*sigma**2) * psi_e**2)
    else:
        return 0


def r_pos_e(pos_e):
    """
    r = Ce^{-1/(2*sigma^2) pos_e^2}

    """
    # if np.linalg.norm(pos_e) <= 10:
    sigma = 5   # [m]
    C = 1       # Max. along axis reward

    r1 = C*np.exp(-1/(2*sigma**2) * pos_e[0]**2)
    r2 = C*np.exp(-1/(2*sigma**2) * pos_e[1]**2)
    return r1 + r2


def r_surge(obs):
    """
    Possibility to use surge rewards for docking
    """
    if np.linalg.norm(obs[0:2], 2) <= 5:
        return 0
    else:
        # If u < 0, give negative reward
        if obs[3] < 0:
            return -1
        else:
            return 0


def r_time():
    return -1


def r_gaussian(obs):
    """
    r = C1e^{-1/(2*sigma^2) x_e^2} + C1e^{-1/(2*sigma^2) y_e^2} + C2e^{-1/((2*sigma^2) y_e^2}

    """
    pos_e = np.array([obs[0], obs[1]])

    return r_pos_e(pos_e) + r_psi_e(obs[2], pos_e)
