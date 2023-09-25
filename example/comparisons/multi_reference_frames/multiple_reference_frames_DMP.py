import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
# from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from policy_transportation import GaussianProcess as GPR
from policy_transportation import GaussianProcessTransportation as Transport
from matplotlib.patches import Circle
from policy_transportation.plot_utils import draw_error_band
import warnings
import os
import similaritymeasures
import random
warnings.filterwarnings("ignore")

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


# distribution_new=np.zeros((len(demos_x),len(demos_x[0])*len(demos_x[0]),2))
# distribution=np.zeros((len(demos_x),4,2))
# distribution_new=np.zeros((len(demos_x),4,2))
distribution=np.zeros((len(demos_x),10,2))
distribution_new=np.zeros((len(demos_x),10,2))
final_distance=np.zeros((len(demos_x),2))
final_orientation=np.zeros((len(demos_x),1))
# index=2
frame_dim=5
for i in range(len(demos_x)):
    distribution[i,0,:]=demos_b[i][0][0]
    distribution[i,1,:]=demos_b[i][0][0]+demos_A[i][0][0] @ np.array([ 0, frame_dim])
    distribution[i,2,:]=demos_b[i][0][1]
    distribution[i,3,:]=demos_b[i][0][1]+demos_A[i][0][1] @ np.array([ 0, -frame_dim])
    # Extra points
    distribution[i,4,:]=demos_b[i][0][0]+demos_A[i][0][0] @ np.array([ 0, -frame_dim])
    distribution[i,5,:]=demos_b[i][0][1]+demos_A[i][0][1] @ np.array([ 0, frame_dim])

    distribution[i,6,:]=demos_b[i][0][0]+demos_A[i][0][0] @ np.array([ +frame_dim, 0])
    distribution[i,7,:]=demos_b[i][0][1]+demos_A[i][0][1] @ np.array([ +frame_dim, 0])
    distribution[i,8,:]=demos_b[i][0][0]+demos_A[i][0][0] @ np.array([ -frame_dim, 0])
    distribution[i,9,:]=demos_b[i][0][1]+demos_A[i][0][1] @ np.array([ -frame_dim, 0])


    final_distance[i]=  np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,:] - demos_b[i][0][1])

    final_delta=np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,:]-demos_x[i][-2,:])
    final_orientation[i]= np.arctan2(final_delta[1],final_delta[0])



demos_A_new=np.load('demos_A.npy', allow_pickle=True)
demos_b_new=np.load('demos_b.npy', allow_pickle=True)

for i in range(len(demos_x)):
    distribution_new[i,0,:]=demos_b_new[i][0][0]
    distribution_new[i,1,:]=demos_b_new[i][0][0]+demos_A_new[i][0][0] @ np.array([ 0, frame_dim])
    distribution_new[i,2,:]=demos_b_new[i][0][1]
    distribution_new[i,3,:]=demos_b_new[i][0][1]+demos_A_new[i][0][1] @ np.array([ 0, -frame_dim])
    # #Extra points
    distribution_new[i,4,:]=demos_b_new[i][0][0]+demos_A_new[i][0][0] @ np.array([ 0, -frame_dim])
    distribution_new[i,5,:]=demos_b_new[i][0][1]+demos_A_new[i][0][1] @ np.array([ 0, frame_dim])

    distribution_new[i,6,:]=demos_b_new[i][0][0]+demos_A_new[i][0][0] @ np.array([ +frame_dim, 0])
    distribution_new[i,7,:]=demos_b_new[i][0][1]+demos_A_new[i][0][1] @ np.array([ +frame_dim, 0])
    distribution_new[i,8,:]=demos_b_new[i][0][0]+demos_A_new[i][0][0] @ np.array([ -frame_dim, 0])
    distribution_new[i,9,:]=demos_b_new[i][0][1]+demos_A_new[i][0][1] @ np.array([ -frame_dim, 0])
    

def execute(distribution_input, index_source, index_target, plot=True, training_set=False):

    X=demos_x[index_source]
    transport=Transport()
    transport.source_distribution=distribution[index_source,:,:]
    transport.target_distribution=distribution_input[index_target,:,:]
    transport.training_traj=X
    transport.fit_transportation_linear()
    transport.apply_transportation_linear()
    X1=transport.training_traj
    std=np.zeros_like(X1)
    std=std

    if plot==True:
        draw_error_band(ax, X1[:,0], X1[:,1], err=std, facecolor= [255.0/256.0,140.0/256.0,0.0], edgecolor="none", alpha=.8)
        ax.plot(transport.target_distribution[0:2,0],transport.target_distribution[0:2,1], linewidth=10, alpha=0.9, c='green')
        ax.scatter(transport.target_distribution[0,0],transport.target_distribution[0,1], linewidth=10, alpha=0.9, c='green')
        ax.plot(transport.target_distribution[2:4,0],transport.target_distribution[2:4,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
        ax.scatter(transport.target_distribution[2,0],transport.target_distribution[2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
        ax.plot(transport.target_distribution[:,0],transport.target_distribution[:,1], 'b*',  linewidth=0.2)
        ax.plot(X1[:,0],X1[:,1], c= [255.0/256.0,20.0/256.0,147.0/256.0])
        if training_set==True:
            ax.plot(demos_x[index_target][:,0],demos_x[index_target][:,1], 'b--')
    # Discrete Frechet distance
    if training_set==True:
        df = similaritymeasures.frechet_dist(demos_x[index_target], X1)

        # quantify the difference between the two curves using
        # area between two curves
        area = similaritymeasures.area_between_two_curves(demos_x[index_target], X1)

        # quantify the difference between the two curves using
        # Dynamic Time Warping distance
        dtw, d = similaritymeasures.dtw(demos_x[index_target], X1)

        fd=  np.linalg.inv(demos_A[index_target][0][1]) @ (X1[-1] - demos_b[index_target][0][1])
        fde=np.linalg.norm(final_distance[index_target]-fd)

        final_vel=  np.linalg.inv(demos_A[index_target][0][1]) @ (X1[-1] - X1[-5])

        final_angle= np.arctan2(final_vel[1], final_vel[0])

        final_angle_distance= np.abs(final_angle - final_orientation[i])  

        print("Final Point Distance  : ", fde)
        print("Frechet Distance      : ", df)
        print("Area between two curves: ", area)
        print("Dynamic Time Warping  : ", dtw)
        print("Final Angle Distance  : ", final_angle_distance[0])
        return df, area, dtw, fde, final_angle_distance[0]
    else:
        fd=  np.linalg.inv(demos_A_new[index_target][0][1]) @ (X1[-1] - demos_b_new[index_target][0][1])
        fde=np.linalg.norm(final_distance[index_target]-fd)

        final_vel=  np.linalg.inv(demos_A_new[index_target][0][1]) @ (X1[-1] - X1[-5])

        final_angle= np.arctan2(final_vel[1], final_vel[0])

        final_angle_distance= np.abs(final_angle - final_orientation[i])

        print("Final Point Distance  : ", fde)
        print("Final Angle Distance  : ", final_angle_distance[0])
        return fde, final_angle_distance[0]



index_source=random.choice(range(9))
print(index_source)
fig, ax = plt.subplots()
for i in range(9):
    execute(distribution, index_source, index_target=i , plot=True, training_set=True)
plt.plot(demos_x[index_source][:,0],demos_x[index_source][:,1], color=[1,0,0]) 
ax.grid(color='gray', linestyle='-', linewidth=1)
# Customize the background color
ax.set_facecolor('white')
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_aspect('equal')
# Customize the background color
ax.set_facecolor('white')
# Remove the box around the figure
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fig.savefig('figs/transportation.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots()

for i in range(9):
    execute(distribution_new, index_source, index_target=i , plot=True, training_set=False)    

ax.grid(color='gray', linestyle='-', linewidth=1)
# Customize the background color
ax.set_facecolor('white')
# ax.set_xlim(-75, 105)
# ax.set_ylim(-95, 70)
# Customize the background color
ax.set_aspect('equal')
ax.set_facecolor('white')
# Remove the box around the figure
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fig.savefig('figs/transportation_new.png', dpi=1200, bbox_inches='tight')
plt.show()


# EXPERIMENTS
number_repetitions=20
results_df=[]
results_area=[]
results_dtw=[]
results_fde=[]
results_fda=[]

#Sample a demonstration at random and generalize on all the avaialable frames. Repeat this 20 times
for j in range(number_repetitions):
    # index_source=random.choice(range(len(demos_x)))
    # vector = np.delete(range(len(demos_x)), index_source) 

    index_source = random.choice(range(len(demos_x)))  # Randomly choose an index from 0 to 9
    vector = [i for i in range(len(demos_x)) if i != index_source] 
    for k in vector:
        df, area, dtw, fde, fda= execute( distribution, index_source, index_target=k , plot=False, training_set=True)
        results_df.append(df)
        results_area.append(area)
        results_dtw.append(dtw)
        results_fde.append(fde)
        results_fda.append(fda)


np.savez('results_affine.npz', 
    results_df=results_df, 
    results_area=results_area, 
    results_dtw=results_dtw,
    results_fde=results_fde, 
    results_fad=results_fda )

# Test on unknown data 
from generate_random_frame_orientation import generate_frame_orientation

results_fde_new=[]
results_fda_new=[]
#Sample a demonstration at random and generalize on all the avaialable frames. Repeat this 20 times
for j in range(number_repetitions):
    demos_A_new, demos_b_new = generate_frame_orientation()
    for demo_i in range(len(demos_x)):
        distribution_new[demo_i,0,:]=demos_b_new[demo_i][0][0]
        distribution_new[demo_i,1,:]=demos_b_new[demo_i][0][0]+demos_A_new[demo_i][0][0] @ np.array([ 0, frame_dim])
        distribution_new[demo_i,2,:]=demos_b_new[demo_i][0][1]
        distribution_new[demo_i,3,:]=demos_b_new[demo_i][0][1]+demos_A_new[demo_i][0][1] @ np.array([ 0, -frame_dim])
        #Extra points
        distribution_new[demo_i,4,:]=demos_b_new[demo_i][0][0]+demos_A_new[demo_i][0][0] @ np.array([ 0, -frame_dim])
        distribution_new[demo_i,5,:]=demos_b_new[demo_i][0][1]+demos_A_new[demo_i][0][1] @ np.array([ 0, frame_dim])

        distribution_new[demo_i,6,:]=demos_b_new[demo_i][0][0]+demos_A_new[demo_i][0][0] @ np.array([ +frame_dim, 0])
        distribution_new[demo_i,7,:]=demos_b_new[demo_i][0][1]+demos_A_new[demo_i][0][1] @ np.array([ +frame_dim, 0])
        distribution_new[demo_i,8,:]=demos_b_new[demo_i][0][0]+demos_A_new[demo_i][0][0] @ np.array([ -frame_dim, 0])
        distribution_new[demo_i,9,:]=demos_b_new[demo_i][0][1]+demos_A_new[demo_i][0][1] @ np.array([ -frame_dim, 0])


    index_source=random.choice(range(len(demos_x)))
    
    for k in range(len(demos_x)):
        fde, fda= execute( distribution_new, index_source, index_target=k , plot=False, training_set=False)
        results_fde_new.append(fde)
        results_fda_new.append(fda)


np.savez('results_affine_out_distribution.npz', 

    results_fde=results_fde_new, 
    results_fad=results_fda_new)


