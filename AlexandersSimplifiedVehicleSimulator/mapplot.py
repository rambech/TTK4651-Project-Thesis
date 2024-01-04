import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D
import numpy as np

# Latex settings for plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{lmodern,amsmath,amsfonts}')

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")

pos = (0, 0)
psi = np.pi/8
file_name = "docking_scenario"


def otter(pos, psi):
    sequence = [[-0.4, 1], [-0.3, 0.8], [-0.3, 0.6], [0.3, 0.6], [0.3, 0.8],
                [0.4, 1], [0.5, 0.8], [0.5, -0.8],
                [0.4, -1], [0.3, -0.8], [0.3, -0.6], [-0.3, -0.6], [-0.3, -0.8],
                [-0.4, -1], [-0.5, -0.8], [-0.5, 0.8]]
    rotation = Affine2D().rotate(psi)
    translation = Affine2D().translate(pos[0], pos[1])
    boat = patches.Polygon(
        sequence, closed=True, edgecolor='#90552a', facecolor='#f4ac67', linewidth=0.5)
    transform = rotation + translation + ax.transData
    boat.set_transform(transform)
    return boat
    # ax.add_patch(boat)


background = patches.Rectangle(
    (-20, -20), 40, 40, edgecolor='#97d2d4', facecolor='#97d2d4', linewidth=1)
dock = patches.Rectangle((-20, 20-5-0.75), 40, 0.75+5,
                         edgecolor='#808080', facecolor='#e6e6e6', linewidth=1)
quay = patches.Rectangle((-2.5, 15-0.75), 5, 2,
                         edgecolor='#00509e', facecolor='#3e628a', linewidth=1, linestyle="-", alpha=0.3)
restricted0 = patches.Rectangle((-15, 15-0.75), 12.5, 2,
                                edgecolor='#595959', facecolor='#000000', linewidth=1, linestyle="-", alpha=0.3)
restricted1 = patches.Rectangle((2.5, 15-0.75), 12.5, 2,
                                edgecolor='#595959', facecolor='#000000', linewidth=1, linestyle="-", alpha=0.3)
bounds = patches.Rectangle(
    (-15, -15), 30, 30, edgecolor="r", facecolor="none", linewidth=1, linestyle="--")

ax.add_patch(background)
ax.add_patch(dock)
ax.add_patch(restricted0)
rest = ax.add_patch(restricted1)
q = ax.add_patch(quay)
b = ax.add_patch(bounds)
asv = ax.add_patch(otter(pos, psi))
# TODO: Add path by simply making a path and adding boat-patches along it

ax.legend([q, rest, b, otter((0, 0), np.pi/2)], ["Permitted area",
          "Restricted area", r'$\mathbb{S}$', "ASV"])

ax.set(xlim=(-20, 20), ylim=(-20, 20),
       xlabel='E', ylabel='N')

if False:
    ax.text(13, 5, r'$\mathbb{S}$', fontsize=12)

plt.savefig(f'figures/{file_name}.pdf', bbox_inches='tight')
plt.show()
