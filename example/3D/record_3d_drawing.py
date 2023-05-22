#%%
from interface_spacenav import Spacenav
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time
%matplotlib qt5

surface = Spacenav()
surface.connect_ROS()
time.sleep(1)




surface.plot_traj()

# %%
