import os 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from models.model_gpt import Multiple_Reference_Frames_GPT
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from policy_transportation import GaussianProcess as GPR
from policy_transportation.plot_utils import plot_vector_field
from policy_transportation.utils import from_list_trajectory_to_array
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module=r"matplotlib\..*")
np.set_printoptions(precision=2) 

script_path = str(os.path.dirname(__file__))
filename = script_path + '/data/' + 'reach_target'

policy=Multiple_Reference_Frames_GPT()
policy.load_dataset(filename)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

def plot_stream_plot(ax, X, deltaX, transparency=0.5, xlim=None, ylim=None):
    kernel_policy = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01) 
    model=GPR(kernel=kernel_policy)
    model.fit(X, deltaX)
    if xlim is None:
        xlim = [np.min(X[:,0]-10), np.max(X[:,0]+10)]
    if ylim is None:
        ylim = [np.min(X[:,1]-10), np.max(X[:,1]+10)]
    x_grid = np.linspace(xlim[0], xlim[1], 100)
    y_grid = np.linspace(ylim[0], ylim[1], 100)
    dataXX, dataYY = np.meshgrid(x_grid, y_grid)
    pos_array= np.column_stack((dataXX.ravel(), dataYY.ravel()))
    [vel, std]=model.predict(pos_array, return_std=True)
    grad=model.derivative_of_variance(pos_array).transpose()
    vel=vel-2*std*grad/np.linalg.norm(grad, axis=1).reshape(-1,1)
    u= vel[:,0].reshape(dataXX.shape)
    v= vel[:,1].reshape(dataXX.shape)
    # Get the streamplot object
    stream = ax.streamplot(dataXX, dataYY, u, v, color='gray')
    # Set alpha (transparency)
    for collection in stream.lines.get_children():  
        collection.set_alpha(transparency)
for target_index in range(8):
    ax = axes[target_index]
    ax.grid(color='gray', linestyle='-', linewidth=1)
    ax.set_facecolor('white')
    xlim = [-60, 60]
    ylim = [-60, 60]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True)
    trajectories = []
    source_index = [i for i in range(8) if i != target_index]
    for i in range(len(source_index)):
        policy.reproduce(source_index[i], target_index, ax=ax, compute_metrics=False, plot_bounds=False)
        position = policy.demos_x[source_index[i]]
        policy.transport.fit(
            policy.distribution_training_set[source_index[i], :, :],
            policy.distribution_training_set[target_index, :, :],
            do_scale=True,
            do_rotation=True
        )
        position_hat = policy.transport.transport(position, return_std=False)
        trajectories.append(position_hat)
    X, delta_X = from_list_trajectory_to_array(trajectories)
    plot_stream_plot(ax, X, delta_X, transparency=0.2, xlim=xlim, ylim=ylim)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(script_path + '/figs/multi_source_single_target_transportation.pdf', dpi=1200, bbox_inches='tight')

plt.show()



