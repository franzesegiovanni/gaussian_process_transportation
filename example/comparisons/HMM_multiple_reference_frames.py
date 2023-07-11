import os 
import numpy as np
import matplotlib.pyplot as plt
import pbdlib as pbd
import similaritymeasures
from pbdlib.utils.jupyter_utils import *
import random
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2)

def random_rotation_matrix_2d():
    theta = np.random.uniform(-np.pi, np.pi)  # Random angle between 0 and 2*pi

    c = np.cos(theta)
    s = np.sin(theta)

    rotation_matrix = np.array([[c, -s],
                                [s, c]])

    return rotation_matrix

filename = 'reach_target'

pbd_path = os. getcwd()  + '/data_hmm/'

demos = np.load(pbd_path + filename + '.npy', allow_pickle=True, encoding='latin1')[()]


### Trajectory data
demos_x = demos['x'] # position
demos_dx = demos['dx'] # velocity
demos_xdx = [np.concatenate([x, dx], axis=1) for x, dx in zip(demos_x, demos_dx)] # concatenation

### Coordinate systems transformation
demos_A = [d for d in demos['A']]
demos_b = [d for d in demos['b']]

### Coordinate systems transformation for concatenation of position-velocity
demos_A_xdx = [np.kron(np.eye(2), d) for d in demos_A]
demos_b_xdx = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b]

### Stacked demonstrations
data_x = np.concatenate([d for d in demos_x], axis=0)

ylim = [np.min(data_x[:, 1]) - 20., np.max(data_x[:, 1]) + 20]
xlim = [np.min(data_x[:, 0]) - 20., np.max(data_x[:, 0]) + 20]

# a new axis is created for the different coordinate systems 
demos_xdx_f = [np.einsum('taji,taj->tai',_A, _x[:, None] - _b) 
               for _x, _A, _b in zip(demos_xdx, demos_A_xdx, demos_b_xdx)] 
# t : timestep, a coordinate systems, i, j : dimensions

# concatenated version of the coordinate systems
demos_xdx_augm = [d.reshape(-1, 8) for d in demos_xdx_f]



# print(demos_xdx_augm[0].shape, demos_xdx_f[0].shape)

distribution=np.zeros((len(demos_x),4,2))
distribution_new=np.zeros((len(demos_x),4,2))
index=2
for i in range(len(demos_x)):
    distribution[i,0,:]=demos_b[i][0][0]
    distribution[i,1,:]=demos_b[i][0][0]+demos_A[i][0][0] @ np.array([ 0, 10])
    distribution[i,2,:]=demos_b[i][0][1]
    distribution[i,3,:]=demos_b[i][0][1]+demos_A[i][0][1] @ np.array([ 0, -10])

demos_A_new=np.load('demos_A.npy', allow_pickle=True)
demos_b_new=np.load('demos_b.npy', allow_pickle=True)

for i in range(len(demos_x)):
    distribution_new[i,0,:]=demos_b_new[i][0][0]
    distribution_new[i,1,:]=demos_b_new[i][0][0]+demos_A_new[i][0][0] @ np.array([ 0, 10])
    distribution_new[i,2,:]=demos_b_new[i][0][1]
    distribution_new[i,3,:]=demos_b_new[i][0][1]+demos_A_new[i][0][1] @ np.array([ 0, -10])

demos_A_xdx_new = [np.kron(np.eye(2), d) for d in demos_A_new]
demos_b_xdx_new = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b_new]

model = pbd.HMM(nb_states=5, nb_dim=8) # nb_states is the number of Gaussian components


sampled_demo=random.sample(demos_xdx_augm, 9)
indices = [demos_xdx_augm.index(element) for element in sampled_demo]
model.init_hmm_kbins(sampled_demo) # initializing model

# EM to train model
model.em(sampled_demo, reg=1e-3) 

def execute(i, A, b, distribution, start, plot=True, training_set=False):

    _mod1 = model.marginal_model(slice(0, 4)).lintrans(A[0], b[0])
    _mod2 = model.marginal_model(slice(4, 8)).lintrans(A[1], b[1])
    
    # product 
    _prod = _mod1 * _mod2
    
    # get the most probable sequence of state for this demonstration
    sq = model.viterbi(demos_xdx_augm[i])
    
    # solving LQR with Product of Gaussian, see notebook on LQR
    lqr = pbd.PoGLQR(nb_dim=2, dt=0.05, horizon=demos_xdx[i].shape[0])
    lqr.mvn_xi = _prod.concatenate_gaussian(sq) # augmented version of gaussian
    lqr.mvn_u = -4.
    lqr.x0 = start
    
    xi = lqr.seq_xi
    ax.plot(xi[:, 0], xi[:, 1], color=[255.0/256.0,20.0/256.0,147.0/256.0], lw=2)
    
    # pbd.plot_gmm(_mod1.mu, _mod1.sigma, swap=True, ax=ax[i], dim=[0, 1], color='steelblue', alpha=0.3)
    # pbd.plot_gmm(_mod2.mu, _mod2.sigma, swap=True, ax=ax[i], dim=[0, 1], color='orangered', alpha=0.3)
    if plot==True:
        pbd.plot_gmm(_prod.mu, _prod.sigma, swap=True, ax=ax, dim=[0, 1], color=[255.0/256.0,140.0/256.0,0.0], alpha=0.5)
        ax.plot(distribution[i][0:2,0],distribution[i][0:2,1], linewidth=10, alpha=0.9, c='green')
        ax.plot(distribution[i][2:4,0],distribution[i][2:4,1], linewidth=10, alpha=0.9, c=[30.0/256.0,144.0/256.0,255.0/256.0])
        ax.scatter(distribution[i][0,0],distribution[i][0,1], linewidth=10, alpha=0.9, c='green')
        ax.scatter(distribution[i][2,0],distribution[i][2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
    if training_set==True:
        ax.plot(demos_x[i][:, 0], demos_x[i][:, 1], 'k--', lw=2)
    
    if training_set==True:
        df = similaritymeasures.frechet_dist(demos_x[i], xi[:,0:2])
        # area between two curves
        area = similaritymeasures.area_between_two_curves(demos_x[i], xi[:,0:2])
        # Dynamic Time Warping distance
        dtw, d = similaritymeasures.dtw(demos_x[i], xi[:,0:2])
        # Final Displacement Error
        fde = np.linalg.norm(demos_x[i][-1] - xi[-1,0:2])
    else:
        df='Nan'
        area='Nan'
        dtw='Nan'    
        fde=np.linalg.norm(distribution[i][2,:]-xi[-1,0:2])
    print("Final Displacement Error: ", fde)
    print("Frechet Distance      : ", df)
    print("Area between two curves: ", area)
    print("Dynamic Time Warping  : ", dtw)
    print("Final Point Distance  : ", fde)

    return df, area, dtw, fde
# plt.tight_layout()
fig, ax = plt.subplots()
for i in range(len(demos_x)):
    A, b = demos_A_xdx[i][0], demos_b_xdx[i][0]
    start=demos_xdx[i][0]
    execute(i, A, b, distribution, start, training_set=True)
# plt.grid('on')
ax.grid(color='gray', linestyle='-', linewidth=0.5)
# Customize the background color
ax.set_facecolor('white')
ax.axis('equal')
ax.patch.set_linewidth(4)
ax.patch.set_edgecolor('black')

fig, ax = plt.subplots()
for i in range(len(demos_x)):
    # A, b = demos_A_xdx[i][0], demos_b_xdx[i][0]
    A, b =demos_A_xdx_new[i][0], demos_b_xdx_new[i][0]
    start=np.zeros_like(demos_xdx[i][0])
    start=demos_xdx[i][0]+(demos_b_xdx_new[i][0][0]-demos_b_xdx[i][0][0])
    vel_new=A[0][2:4,2:4] @ np.linalg.inv(demos_A_xdx_new[i][0][0][2:4,2:4]) @ start[2:]
    start[2:]=vel_new
    execute(i, A, b, distribution_new, start)

ax.grid(color='gray', linestyle='-', linewidth=0.5)
# Customize the background color
ax.set_facecolor('white')
ax.axis('equal')
ax.patch.set_linewidth(4)
ax.patch.set_edgecolor('black')

plt.show()

# Experiments
number_repetitions=20
results_df=np.zeros( ( len(demos_x) ,number_repetitions, len(demos_x)) )
results_area=np.zeros(( len(demos_x) ,number_repetitions, len(demos_x)) )
results_dtw=np.zeros(( len(demos_x) , number_repetitions, len(demos_x) ))
results_fde=np.zeros((  len(demos_x) , number_repetitions , len(demos_x) ))

for i in range(len(demos_xdx_augm)):
    for j in range(number_repetitions):
        sampled_demo=random.sample(demos_xdx_augm, i+1)
        indices = [demos_xdx_augm.index(element) for element in sampled_demo]
        model.init_hmm_kbins(sampled_demo) # initializing model

        # EM to train model
        model.em(sampled_demo, reg=1e-3) 

        for k in range(len(demos_x)):
            A, b = demos_A_xdx[k][0], demos_b_xdx[k][0]
            start=demos_xdx[k][0]
            results_df[i,j,k], results_area[i,j,k], results_dtw[i,j,k], results_fde[i,j,k]= execute(k, A, b, distribution, start, plot=False, training_set=True)


np.savez('results_sota_dataset.npz', 
    results_df=results_df, 
    results_area=results_area, 
    results_dtw=results_dtw,
    results_fde=results_fde)

i=0
results_df=np.zeros( ( number_repetitions, len(demos_x)) )
results_area=np.zeros(( number_repetitions, len(demos_x)) )
results_dtw=np.zeros(( number_repetitions, len(demos_x) ))
results_fde=np.zeros((  number_repetitions , len(demos_x) ))

for j in range(number_repetitions):
    sampled_demo=random.sample(demos_xdx_augm, 9)
    # indices = [demos_xdx_augm.index(element) for element in sampled_demo]
    model.init_hmm_kbins(sampled_demo) # initializing model

    # EM to train model
    model.em(sampled_demo, reg=1e-3) 

    for k in range(len(demos_x)):
            # A, b = demos_A_xdx[i][0], demos_b_xdx[i][0]
        A, b = demos_A_xdx[k][0], demos_b_xdx[k][0]
        start=np.zeros_like(demos_xdx[k][0])
        start=demos_xdx[k][0]+(demos_b_xdx_new[k][0][0]-demos_b_xdx[k][0][0])
        vel_new=A[0][2:4,2:4] @ np.linalg.inv(demos_A_xdx_new[i][0][0][2:4,2:4]) @ start[2:]
        start[2:]=vel_new

        _, _, _, results_fde[j,k]= execute(k, A, b, distribution, start, plot=False, training_set=False)


np.savez('results_sota_new_points.npz', 
    results_fde=results_fde)


    