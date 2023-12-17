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
