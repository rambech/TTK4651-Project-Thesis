import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D
import numpy as np

from rl.rewards import r_heading

# Latex settings for plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')

fig, ax = plt.subplots(figsize=(5, 5))
# ax.set_aspect("equal")

file_name = "1d_gaussian"

if True:
    sigma = np.pi/8    # [rad]
    C = 0.5            # Max. along axis reward

    # x = np.arange(-np.pi/2, np.pi/2, 1)
    # y = C*np.exp(-(1/(2*sigma**2)) * x**2)

    x = np.arange(-np.pi/2, np.pi/2, 0.01)  # Adjust the step size as needed
    y = C * np.exp(-(1/(2*sigma**2)) * x**2)

    ax.plot(x, y, color="#2e7578")
    ax.set(xlim=(-np.pi/2, np.pi/2), ylim=(0, 0.5),
           xlabel=r'$\tilde{\psi}$', ylabel='R')

# ax.set(xlim=(-20, 20), ylim=(-20, 20),
#        xlabel='E', ylabel='N')

plt.savefig(f'figures/{file_name}.pdf', bbox_inches='tight')
plt.show()
