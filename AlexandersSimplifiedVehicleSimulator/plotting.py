import matplotlib.pyplot as plt
from rl.rewards import r_pos_e, r_psi_e
import numpy as np

from mpl_toolkits.mplot3d import axes3d


ax = plt.figure().add_subplot(projection='3d')
# ax1 = plt.figure().add_subplot(projection='3d')
x = y = np.arange(-15, 15, 0.05)
X, Y = np.meshgrid(x, y)
Zer = np.zeros((500, 2), float)
# Z1 = np.zeros((len(X[0]), len(X[1]), 2), float)
Z = r_pos_e((X, Y))

# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor='royalblue', linewidth=0.5, rstride=8, cstride=8,
                alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph.
# ax.contour(X, Y, Z, zdir='z', offset=-3, cmap='coolwarm')
# ax.contour(X, Y, Z, zdir='x', offset=-20, cmap='coolwarm')
# ax.contour(X, Y, Z, zdir='y', offset=20, cmap='coolwarm')

ax.set(xlim=(-16, 16), ylim=(-16, 16), zlim=(-1, 2),
       xlabel='X', ylabel='Y', zlabel='Z')


# Plot the 3D surface
# ax1.plot_surface(X, Y, Z1, edgecolor='seagreen', lw=0.5, rstride=8, cstride=8,
#                  alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph.
# ax1.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
# ax1.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
# ax1.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

# ax1.set(xlim=(-20, 20), ylim=(-20, 20), zlim=(-3, 3),
#         xlabel='X', ylabel='Y', zlabel='Z')

plt.show()
