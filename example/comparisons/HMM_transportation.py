import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
# from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from GILoSA import GaussianProcess as GPR
from GILoSA import Transport
from matplotlib.patches import Circle
from plot_utils import plot_vector_field_minvar, draw_error_band
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
distribution=np.zeros((len(demos_x),4,2))
distribution_new=np.zeros((len(demos_x),4,2))
index=2
for i in range(len(demos_x)):
    distribution[i,0,:]=demos_b[i][0][0]
    distribution[i,1,:]=demos_b[i][0][0]+demos_A[i][0][0] @ np.array([ 0, 10])
    distribution[i,2,:]=demos_b[i][0][1]
    distribution[i,3,:]=demos_b[i][0][1]+demos_A[i][0][1] @ np.array([ 0, -10])


demos_A=np.load('demos_A.npy', allow_pickle=True)
demos_b=np.load('demos_b.npy', allow_pickle=True)

for i in range(len(demos_x)):
    distribution_new[i,0,:]=demos_b[i][0][0]
    distribution_new[i,1,:]=demos_b[i][0][0]+demos_A[i][0][0] @ np.array([ 0, 10])
    distribution_new[i,2,:]=demos_b[i][0][1]
    distribution_new[i,3,:]=demos_b[i][0][1]+demos_A[i][0][1] @ np.array([ 0, -10])


def execute(distribution_input, index_source, index_target, plot=True, training_set=False):

    k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01) #this kernel works much better!    
    gp_deltaX=GPR(kernel=k_deltaX)
    X=demos_x[index_source]
    deltaX=demos_dx[index_source]
    gp_deltaX.fit(X, deltaX)

    transport=Transport()
    transport.source_distribution=distribution[index_source,:,:]
    transport.target_distribution=distribution_input[index_target,:,:]
    transport.training_traj=X
    transport.training_delta=deltaX
    k_transport = C(constant_value=np.sqrt(10))  * RBF(20*np.ones(1), [30,50]) + WhiteKernel(0.01 , [0.0000001, 0.000001])
    transport.kernel_transport=k_transport
    transport.fit_transportation()
    transport.apply_transportation()
    X1=transport.training_traj
    deltaX1=transport.training_delta 
    _,std=transport.gp_delta_map.predict(transport.training_traj)


    k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(1), [30,50], nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
    gp_deltaX1=GPR(kernel=k_deltaX1)
    gp_deltaX1.fit(X1, deltaX1)

    if plot==True:
        draw_error_band(ax, X1[:,0], X1[:,1], err=std[:], facecolor= [255.0/256.0,140.0/256.0,0.0], edgecolor="none", alpha=.8)
        ax.plot(transport.target_distribution[0:2,0],transport.target_distribution[0:2,1], linewidth=10, alpha=0.9, c='green')
        ax.scatter(transport.target_distribution[0,0],transport.target_distribution[0,1], linewidth=10, alpha=0.9, c='green')
        ax.plot(transport.target_distribution[2:4,0],transport.target_distribution[2:4,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
        ax.scatter(transport.target_distribution[2,0],transport.target_distribution[2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
        ax.plot(X1[:,0],X1[:,1], c= [255.0/256.0,20.0/256.0,147.0/256.0])
        if training_set==True:
            ax.plot(demos_x[i][:,0],demos_x[i][:,1], 'b--') 
    # distance=np.linalg.norm(demos_x[i]-X1)
    # print(distance)
    # Discrete Frechet distance
    if training_set==True:
        df = similaritymeasures.frechet_dist(demos_x[i], X1)

        # quantify the difference between the two curves using
        # area between two curves
        area = similaritymeasures.area_between_two_curves(demos_x[i], X1)

        # quantify the difference between the two curves using
        # Dynamic Time Warping distance
        dtw, d = similaritymeasures.dtw(demos_x[i], X1)

        fde=np.linalg.norm(demos_x[i][-1]-X1[-1])
        print("Final Point Distance  : ", fde)
        print("Frechet Distance      : ", df)
        print("Area between two curves: ", area)
        print("Dynamic Time Warping  : ", dtw)
    else:
        fde=np.linalg.norm(transport.target_distribution[2,:]-X1[-1])
        print("Final Point Distance  : ", fde)
        df='Nan'
        dtw='Nan'
        area='Nan'
    return df, area, dtw, fde


index_source=random.choice(range(9))
print(index_source)
fig, ax = plt.subplots()
for i in range(9):
    execute(distribution, index_source, index_target=i , plot=True, training_set=True)
plt.plot(demos_x[index_source][:,0],demos_x[index_source][:,1], color=[1,0,0]) 
ax.axis('equal')   
ax.grid(color='gray', linestyle='-', linewidth=0.5)
# Customize the background color
ax.set_facecolor('white')

fig, ax = plt.subplots()

for i in range(9):
    execute(distribution_new, index_source, index_target=i , plot=True, training_set=False)    
ax.axis('equal')   
ax.grid(color='gray', linestyle='-', linewidth=0.5)
# Customize the background color
ax.set_facecolor('white')
plt.show()


# Experiments
number_repetitions=20
results_df=np.zeros( ( number_repetitions, len(demos_x)) )
results_area=np.zeros(( number_repetitions, len(demos_x)) )
results_dtw=np.zeros(( number_repetitions, len(demos_x) ) )
results_fde=np.zeros((  number_repetitions , len(demos_x) ))

#Sample a demonstration at random and generalize on all the avaialable frames. Repeat this 20 times
for j in range(number_repetitions):
    index_source=random.choice(range(len(demos_x)))
    
    for k in range(len(demos_x)):
        results_df[j,k], results_area[j,k], results_dtw[j,k], results_fde[j,k]= execute( distribution, index_source, index_target=k , plot=False, training_set=True)


np.savez('results_transportation_dataset.npz', 
    results_df=results_df, 
    results_area=results_area, 
    results_dtw=results_dtw,
    results_fde=results_fde)

# Test on unknown data 


#Sample a demonstration at random and generalize on all the avaialable frames. Repeat this 20 times
for j in range(number_repetitions):
    index_source=random.choice(range(len(demos_x)))
    
    for k in range(len(demos_x)):
        _, _, _, results_fde[j,k]= execute( distribution_new, index_source, index_target=k , plot=False, training_set=False)


np.savez('results_transportation.npz', 
    results_df=results_df, 
    results_area=results_area, 
    results_dtw=results_dtw,
    results_fde=results_fde)


