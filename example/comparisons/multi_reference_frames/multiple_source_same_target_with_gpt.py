import os 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from models.model_gpt import Multiple_Reference_Frames_GPT
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from policy_transportation import GaussianProcess as GPR
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2) 

script_path = str(os.path.dirname(__file__))
filename = script_path + '/data/' + 'reach_target'

policy=Multiple_Reference_Frames_GPT()
policy.load_dataset(filename)
fig, ax = plt.subplots()
ax.grid(color='gray', linestyle='-', linewidth=1)
# Customize the background color
ax.set_facecolor('white')
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True)
ax.set_title('Multi Source Single Target Transportation', fontsize=12)
source_index=[0,1,2,4,5,6,7,8]
target_index=3
traj_list=[]
X=np.empty((0,2))
dX=np.empty((0,2))
for i in range(len(source_index)):
    policy.reproduce(source_index[i], target_index, ax=ax, compute_metrics=False,plot_bounds=False)
    X=np.vstack((X, policy.transport.training_traj))
    deltaX=np.zeros((len(policy.transport.training_traj),2))
    for j in range(len(policy.transport.training_traj)-1):
        deltaX[j,:]=(policy.transport.training_traj[j+1,:]-policy.transport.training_traj[j,:])
    dX=np.vstack((dX, deltaX))    
# policy.reproduce(source_index, target_index, ax=ax, compute_metrics=False, plot_bounds=False)

for spine in ax.spines.values():
    spine.set_linewidth(2)

# plt.show()
#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=0.5) + WhiteKernel(0.01) 
model=GPR(kernel=k_deltaX)
model.fit(X, dX)
x_grid=np.linspace(-60, 60, 100)
y_grid=np.linspace(-60, 60, 100)

dataXX, dataYY = np.meshgrid(x_grid, y_grid)
u=np.ones((len(x_grid),len(y_grid)))
v=np.ones((len(x_grid),len(y_grid)))
for i in range(len(x_grid)):
    for j in range(len(y_grid)):
        pos=np.array([dataXX[i,j],dataYY[i, j]]).reshape(1,-1)
        [vel, std]=model.predict(pos)
        [_,grad]=model.derivative(pos)
        u[i,j]=vel[0,0]-3*std[0][0]*grad[0,0,0]/np.sqrt(grad[0,0,0]**2+grad[0,0,0]**2)
        v[i,j]=vel[0,1]-3*std[0][0]*grad[0,1,0]/np.sqrt(grad[0,0,0]**2+grad[0,1,0]**2)
plt.streamplot(dataXX, dataYY, u, v, density = 2, color='c', linewidth=1)

fig.savefig(script_path + '/figs/multisourcesingletarget.pdf', dpi=1200, bbox_inches='tight')

plt.show()



