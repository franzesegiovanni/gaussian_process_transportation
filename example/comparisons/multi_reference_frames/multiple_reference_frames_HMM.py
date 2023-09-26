import os 
import numpy as np
import matplotlib.pyplot as plt
import pbdlib as pbd
import similaritymeasures
from pbdlib.utils.jupyter_utils import *
import random
import warnings
import pickle
from generate_random_frame_orientation import generate_frame_orientation
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

def execute(i, A, b, distribution, start, plot=True, training_set=False):

    _mod1 = model.marginal_model(slice(0, 4)).lintrans(A[0], b[0])
    _mod2 = model.marginal_model(slice(4, 8)).lintrans(A[1], b[1])
    
    # product 
    _prod = _mod1 * _mod2
    
    # get the most probable sequence of state for this demonstration
    if training_set:
        sq = model.viterbi(demos_xdx_augm[i])
        lqr = pbd.PoGLQR(nb_dim=2, dt=0.05, horizon=demos_xdx[i].shape[0])
    else:
        sq = [int(count // (horizon_mean/nb_states)) for count in range(horizon_mean)]
        lqr = pbd.PoGLQR(nb_dim=2, dt=0.05, horizon=horizon_mean)       
    #sq = [i // (n/m) for i in range(n)]
    # solving LQR with Product of Gaussian, see notebook on LQR
    

    lqr.mvn_xi = _prod.concatenate_gaussian(sq) # augmented version of gaussian
    lqr.mvn_u = -4.
    lqr.x0 = start
    
    xi = lqr.seq_xi
    ax.plot(xi[:, 0], xi[:, 1], color=[255.0/256.0,20.0/256.0,147.0/256.0], lw=2)
    
    # pbd.plot_gmm(_mod1.mu, _mod1.sigma, swap=True, ax=ax[i], dim=[0, 1], color='steelblue', alpha=0.3)
    # pbd.plot_gmm(_mod2.mu, _mod2.sigma, swap=True, ax=ax[i], dim=[0, 1], color='orangered', alpha=0.3)
    if plot==True:
        # pbd.plot_gmm(_prod.mu, _prod.sigma, swap=True, ax=ax, dim=[0, 1], color=[255.0/256.0,140.0/256.0,0.0], alpha=0.5)
        ax.plot(distribution[i][0:2,0],distribution[i][0:2,1], linewidth=10, alpha=0.9, c='green')
        ax.plot(distribution[i][2:4,0],distribution[i][2:4,1], linewidth=10, alpha=0.9, c=[30.0/256.0,144.0/256.0,255.0/256.0])
        ax.scatter(distribution[i][0,0],distribution[i][0,1], linewidth=10, alpha=0.9, c='green')
        ax.scatter(distribution[i][2,0],distribution[i][2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])

    if training_set==True:
        ax.plot(demos_x[i][:, 0], demos_x[i][:, 1], 'k--', lw=2)
        df = similaritymeasures.frechet_dist(demos_x[i], xi[:,0:2])
        # area between two curves
        area = similaritymeasures.area_between_two_curves(demos_x[i], xi[:,0:2])
        # Dynamic Time Warping distance
        dtw, d = similaritymeasures.dtw(demos_x[i], xi[:,0:2])
        # Final Displacement Error
        # fde = np.linalg.norm(demos_x[i][-1] - xi[-1,0:2])
        fd=  np.linalg.inv(A[1][0:2,0:2]) @ (xi[-1, 0:2] - b[1][0:2])
        fde=np.linalg.norm(final_distance[i]-fd)

        final_vel=  np.linalg.inv(A[1][0:2,0:2]) @ (xi[-1, 0:2] - xi[-5, 0:2])

        final_angle= np.arctan2(final_vel[1], final_vel[0])

        final_angle_distance= np.abs(final_angle - final_orientation[i])

        print("Final Displacement Error: ", fde)
        print("Frechet Distance      : ", df)
        print("Area between two curves: ", area)
        print("Dynamic Time Warping  : ", dtw)
        print("Final Angle Distance  : ", final_angle_distance[0])
        return df, area, dtw, fde, final_angle_distance[0]
       
    else:   
        # fde=np.linalg.norm(distribution[i][2,:]-xi[-1,0:2])
        fd=  np.linalg.inv(A[1][0:2,0:2]) @ (xi[-1, 0:2] - b[1][0:2])
        fde=np.linalg.norm(final_distance[i]-fd)

        final_vel=  np.linalg.inv(A[1][0:2,0:2]) @ (xi[-1, 0:2] - xi[-5, 0:2])

        final_angle= np.arctan2(final_vel[1], final_vel[0])

        final_angle_distance= np.abs(final_angle - final_orientation[i])

        print("Final Point Distance  : ", fde)
        print("Final Angle Distance  : ", final_angle_distance[0])

        return fde, final_angle_distance[0]



if __name__ == "__main__":
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

    # concatenated version of the coordinate systems
    demos_xdx_augm = [d.reshape(-1, 8) for d in demos_xdx_f]


    distribution=np.zeros((len(demos_x),4,2))
    distribution_new=np.zeros((len(demos_x),4,2))
    final_distance=np.zeros((len(demos_x),2))
    final_orientation=np.zeros((len(demos_x),1))
    index=2
    for i in range(len(demos_x)):
        distribution[i,0,:]=demos_b[i][0][0]
        distribution[i,1,:]=demos_b[i][0][0]+demos_A[i][0][0] @ np.array([ 0, 10])
        distribution[i,2,:]=demos_b[i][0][1]
        distribution[i,3,:]=demos_b[i][0][1]+demos_A[i][0][1] @ np.array([ 0, -10])
        final_distance[i]=  np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,:] - demos_b[i][0][1])
        final_delta=np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,:]-demos_x[i][-2,:])
        final_orientation[i]= np.arctan2(final_delta[1],final_delta[0])

    demos_A_new=np.load('demos_A.npy', allow_pickle=True)
    demos_b_new=np.load('demos_b.npy', allow_pickle=True)

    for i in range(len(demos_x)):
        distribution_new[i,0,:]=demos_b_new[i][0][0]
        distribution_new[i,1,:]=demos_b_new[i][0][0]+demos_A_new[i][0][0] @ np.array([ 0, 10])
        distribution_new[i,2,:]=demos_b_new[i][0][1]
        distribution_new[i,3,:]=demos_b_new[i][0][1]+demos_A_new[i][0][1] @ np.array([ 0, -10])

    demos_A_xdx_new = [np.kron(np.eye(2), d) for d in demos_A_new]
    demos_b_xdx_new = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b_new]

    nb_states = 5
    model = pbd.HMM(nb_states=nb_states, nb_dim=8) # nb_states is the number of Gaussian components

    horizon_mean=0
    for i in range(len(demos_xdx)):
        horizon_mean=np.max([horizon_mean, demos_xdx[i].shape[0]])

    horizon_mean=horizon_mean
    print(horizon_mean)
    # horizon_mean=int(horizon_mean/len(demos_xdx))
    # horizon_mean=200
    sampled_demo=random.sample(demos_xdx_augm, 9)
    indices = [demos_xdx_augm.index(element) for element in sampled_demo]
    model.init_hmm_kbins(sampled_demo) # initializing model

    # EM to train model
    model.em(sampled_demo, reg=1e-3) 

    # plt.tight_layout()
    fig, ax = plt.subplots()
    for i in range(len(demos_x)):
        A, b = demos_A_xdx[i][0], demos_b_xdx[i][0]
        start=demos_xdx[i][0]
        execute(i, A, b, distribution, start, training_set=True)
    # plt.grid('on')
    ax.grid(color='gray', linestyle='-', linewidth=1)
    # Customize the background color
    ax.set_facecolor('white')
    ax.set_xlim(-60, 60)
    ax.set_ylim(-45, 60)
    # ax.patch.set_linewidth(4)
    # ax.patch.set_edgecolor('black')
    fig.savefig('figs/hmm.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots()
    for i in range(len(demos_x)):
        # A, b = demos_A_xdx[i][0], demos_b_xdx[i][0]
        A, b =demos_A_xdx_new[i][0], demos_b_xdx_new[i][0]
        start=np.zeros_like(demos_xdx[i][0])
        start=demos_xdx[i][0]+(demos_b_xdx_new[i][0][0]-demos_b_xdx[i][0][0])
        vel_new=A[0][2:4,2:4] @ np.linalg.inv(demos_A_xdx[i][0][0][2:4,2:4]) @ start[2:]
        start[2:]=vel_new
        execute(i, A, b, distribution_new, start)

    ax.grid(color='gray', linestyle='-', linewidth=1)
    # Customize the background color
    ax.set_facecolor('white')
    # ax.set_xlim(-75, 105)
    # ax.set_ylim(-95, 70)
    # ax.axis('equal')
    # ax.patch.set_linewidth(4)
    # ax.patch.set_edgecolor('black')
    fig.savefig('figs/hmm_new.png', dpi=1200, bbox_inches='tight')

    # plt.show()

    # Experiments
    number_repetitions=20
    demonstrations = len(demos_xdx_augm)-1

    # Create an empty one-dimensional list with demonstrations using list comprehension
    results_df = [[] for _ in range(demonstrations)]
    results_area= [[] for _ in range(demonstrations)]
    results_dtw= [[] for _ in range(demonstrations)]
    results_fde= [[] for _ in range(demonstrations)]
    results_fad= [[] for _ in range(demonstrations)]

    for i in range(demonstrations):
        for j in range(number_repetitions):
            sampled_demo=random.sample(demos_xdx_augm, i+1)
            indices = [demos_xdx_augm.index(element) for element in sampled_demo]
            not_in_indices = [index for index in range(len(demos_xdx_augm)) if index not in indices]
            model.init_hmm_kbins(sampled_demo) # initializing model

            # EM to train model
            model.em(sampled_demo, reg=1e-3) 

            #for k in range(len(demos_x)):
            for k in not_in_indices:    
                A, b = demos_A_xdx[k][0], demos_b_xdx[k][0]
                start=demos_xdx[k][0]
                # results_df[i,j,k], results_area[i,j,k], results_dtw[i,j,k], results_fde[i,j,k]= execute(k, A, b, distribution, start, plot=False, training_set=True)
                df, area, dtw, fde, fad= execute(k, A, b, distribution, start, plot=False, training_set=True)

                results_df[i].append(df)
                results_area[i].append(area)
                results_dtw[i].append(dtw)
                results_fde[i].append(fde)
                results_fad[i].append(fad)


    np.savez('results_sota_dataset.npz', 
        results_df=results_df, 
        results_area=results_area, 
        results_dtw=results_dtw,
        results_fde=results_fde, 
        results_fad=results_fad)

    i=0
    results_df=np.zeros( ( number_repetitions, len(demos_x)) )
    results_area=np.zeros(( number_repetitions, len(demos_x)) )
    results_dtw=np.zeros(( number_repetitions, len(demos_x) ))
    results_fde=np.zeros((  number_repetitions , len(demos_x) ))

    results_fde_new= []
    results_fad_new= []
    #we use always all the demos in the training set and we compute the error to reach the final point in a new situation 
    for j in range(number_repetitions):
        fig, ax = plt.subplots()

        demos_A_new, demos_b_new = generate_frame_orientation()
        for demo_i in range(len(demos_x)):
            distribution_new[demo_i,0,:]=demos_b_new[demo_i][0][0]
            distribution_new[demo_i,1,:]=demos_b_new[demo_i][0][0]+demos_A_new[demo_i][0][0] @ np.array([ 0, 10])
            distribution_new[demo_i,2,:]=demos_b_new[demo_i][0][1]
            distribution_new[demo_i,3,:]=demos_b_new[demo_i][0][1]+demos_A_new[demo_i][0][1] @ np.array([ 0, -10])

        demos_A_xdx_new = [np.kron(np.eye(2), d) for d in demos_A_new]
        demos_b_xdx_new = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b_new]

        sampled_demo=random.sample(demos_xdx_augm, 9)
        # indices = [demos_xdx_augm.index(element) for element in sampled_demo]
        model.init_hmm_kbins(sampled_demo) # initializing model

        # EM to train model
        model.em(sampled_demo, reg=1e-3) 

        for k in range(len(demos_x)):
            A, b = demos_A_xdx_new[k][0], demos_b_xdx_new[k][0]
            start=np.zeros_like(demos_xdx[k][0])
            start=demos_xdx[k][0]+(demos_b_xdx_new[k][0][0]-demos_b_xdx[k][0][0])
            vel_new=A[0][2:4,2:4] @ np.linalg.inv(demos_A_xdx[k][0][0][2:4,2:4]) @ start[2:]
            start[2:]=vel_new

            fde, fad = execute(k, A, b, distribution_new, start, plot=False, training_set=False)
            results_fde_new.append(fde)
            results_fad_new.append(fad)

    # plt.show()
    np.savez('results_sota_out_distribution.npz', 
        results_fde=results_fde_new,
        results_fad=results_fad_new)
    # with open('results_sota_out_distribution.pickle', 'wb') as file:
    #     pickle.dump(results_fde_new, file)


    # with open('results_sota_out_distribution.pickle', 'rb') as file: