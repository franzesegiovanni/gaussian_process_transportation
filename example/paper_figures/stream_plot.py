import matplotlib.pyplot as plt
import numpy as np

fig_tmp, ax_tmp = plt.subplots()
x, y = np.mgrid[0:2.5:1000j, -2.5:2.5:1000j]
vx, vy = np.cos(x - y), np.sin(x - y)
res = ax_tmp.streamplot(x.T, y.T, vx, vy, color='k')
fig_tmp.show()
# extract the lines from the temporary figure
lines = res.lines.get_paths()
#for l in lines:
#    plot(l.vertices.T[0],l.vertices.T[1],'k')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for line in lines:
    old_x = line.vertices.T[0]
    old_y = line.vertices.T[1]
    # apply for 2d to 3d transformation here
    new_z = np.exp(-(old_x ** 2 + old_y ** 2) / 4)
    new_x = 1.2 * old_x
    new_y = 0.8 * old_y
    ax.plot(new_x, new_y, new_z, 'k')
plt.show()