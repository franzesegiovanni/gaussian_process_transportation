import numpy as np
import matplotlib.pyplot as plt
from pbdlib.hmm import HMM
from pbdlib.poglqr import PoGLQR
import similaritymeasures
import random

class Multiple_reference_frames_HMM():
    def __init__(self):
        self.nb_states = 5
        self.model = HMM(nb_states=self.nb_states, nb_dim=8) # nb_states is the number of Gaussian components
        
    def load_data(self, filename):

        demos = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]


        ### Trajectory data
        self.demos_x = demos['x'] # position
        demos_dx = demos['dx'] # velocity
        self.demos_xdx = [np.concatenate([x, dx], axis=1) for x, dx in zip(self.demos_x, demos_dx)] # concatenation

        ### Coordinate systems transformation
        demos_A = [d for d in demos['A']]
        demos_b = [d for d in demos['b']]

        self.demos_A=demos_A
        self.demos_b=demos_b
        ### Coordinate systems transformation for concatenation of position-velocity
        self.demos_A_xdx = [np.kron(np.eye(2), d) for d in demos_A]
        self.demos_b_xdx = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b]

        ### Stacked demonstrations
        data_x = np.concatenate([d for d in self.demos_x], axis=0)

        ylim = [np.min(data_x[:, 1]) - 20., np.max(data_x[:, 1]) + 20]
        xlim = [np.min(data_x[:, 0]) - 20., np.max(data_x[:, 0]) + 20]

        # a new axis is created for the different coordinate systems 
        demos_xdx_f = [np.einsum('taji,taj->tai',_A, _x[:, None] - _b) 
                    for _x, _A, _b in zip(self.demos_xdx, self.demos_A_xdx, self.demos_b_xdx)] 

        # concatenated version of the coordinate systems
        self.demos_xdx_augm = [d.reshape(-1, 8) for d in demos_xdx_f]
        self.demos_xdx_augm_load = [d.reshape(-1, 8) for d in demos_xdx_f]

        # distribution=np.zeros((len(demos_x),4,2))
        # distribution_new=np.zeros((len(demos_x),4,2))
        self.final_distance=np.zeros((len(self.demos_x),2))
        self.final_orientation=np.zeros((len(self.demos_x),1))
        for i in range(len(self.demos_x)):
            self.final_distance[i]=  np.linalg.inv(demos_A[i][0][1]) @ (self.demos_x[i][-1,:] - demos_b[i][0][1])
            final_delta=np.linalg.inv(demos_A[i][0][1]) @ (self.demos_x[i][-1,:]-self.demos_x[i][-2,:])
            self.final_orientation[i]= np.arctan2(final_delta[1],final_delta[0])
    
        horizon_mean=0
        for i in range(len(self.demos_xdx)):
            horizon_mean=np.max([horizon_mean, self.demos_xdx[i].shape[0]])

        self.horizon_mean=horizon_mean

    def train(self):

        self.model.init_hmm_kbins(self.demos_xdx_augm) # initializing model

        # EM to train model
        self.model.em(self.demos_xdx_augm, reg=1e-3) 

    def reproduce(self, index_in_training_set, ax=None, compute_metrics=False):

        A, b = self.demos_A_xdx[index_in_training_set][0], self.demos_b_xdx[index_in_training_set][0]
        start=self.demos_xdx[index_in_training_set][0]
        _mod1 = self.model.marginal_model(slice(0, 4)).lintrans(A[0], b[0])
        _mod2 = self.model.marginal_model(slice(4, 8)).lintrans(A[1], b[1])
        
        # product 
        _prod = _mod1 * _mod2
        
        sq = self.model.viterbi(self.demos_xdx_augm_load[index_in_training_set])
        print('Horizon is ')
        print(self.demos_xdx[index_in_training_set].shape[0])
        lqr = PoGLQR(nb_dim=2, dt=0.05, horizon=self.demos_xdx[index_in_training_set].shape[0])     
        #sq = [i // (n/m) for i in range(n)]
        # solving LQR with Product of Gaussian, see notebook on LQR
        

        lqr.mvn_xi = _prod.concatenate_gaussian(sq) # augmented version of gaussian
        lqr.mvn_u = -4.
        lqr.x0 = start
        
        xi = lqr.seq_xi
        
        
        # pbd.plot_gmm(_mod1.mu, _mod1.sigma, swap=True, ax=ax[index_in_training_set], dim=[0, 1], color='steelblue', alpha=0.3)
        # pbd.plot_gmm(_mod2.mu, _mod2.sigma, swap=True, ax=ax[index_in_training_set], dim=[0, 1], color='orangered', alpha=0.3)
        if ax is not None:
            # ax.grid(color='gray', linestyle='-', linewidth=1)
            # Customize the background color
            # ax.set_facecolor('white')
            # pbd.plot_gmm(_prod.mu, _prod.sigma, swap=True, ax=ax, dim=[0, 1], color=[255.0/256.0,140.0/256.0,0.0], alpha=0.5)
            
            printing_points=np.zeros((4,2))
            print(b[0])
            printing_points[0,:]=b[0][:2]
            printing_points[1,:]=b[0][:2]+A[0][:2,:2] @ np.array([ 0, 10])
            printing_points[2,:]=b[1][:2]
            printing_points[3,:]=b[1][:2]+A[1][:2,:2] @ np.array([ 0, -10])
            ax.plot(xi[:, 0], xi[:, 1], color=[255.0/256.0,20.0/256.0,147.0/256.0], lw=2)
            ax.plot(printing_points[0:2,0],printing_points[0:2,1], linewidth=10, alpha=0.9, c='green')
            ax.plot(printing_points[2:4,0],printing_points[2:4,1], linewidth=10, alpha=0.9, c=[30.0/256.0,144.0/256.0,255.0/256.0])
            ax.scatter(printing_points[0,0],printing_points[0,1], linewidth=10, alpha=0.9, c='green')
            ax.scatter(printing_points[2,0],printing_points[2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])

            

            ax.plot(self.demos_x[index_in_training_set][:, 0], self.demos_x[index_in_training_set][:, 1], 'k--', lw=2)

        if compute_metrics:
            df = similaritymeasures.frechet_dist(self.demos_x[index_in_training_set], xi[:,0:2])
            # area between two curves
            area = similaritymeasures.area_between_two_curves(self.demos_x[index_in_training_set], xi[:,0:2])
            # Dynamic Time Warping distance
            dtw, d = similaritymeasures.dtw(self.demos_x[index_in_training_set], xi[:,0:2])
            # Final Displacement Error
            # fde = np.linalg.norm(demos_x[index_in_training_set][-1] - xi[-1,0:2])
            fd=  np.linalg.inv(A[1][0:2,0:2]) @ (xi[-1, 0:2] - b[1][0:2])
            fde=np.linalg.norm(self.final_distance[index_in_training_set]-fd)

            final_vel=  np.linalg.inv(A[1][0:2,0:2]) @ (xi[-1, 0:2] - xi[-5, 0:2])

            final_angle= np.arctan2(final_vel[1], final_vel[0])

            final_angle_distance= np.abs(final_angle - self.final_orientation[index_in_training_set])

            print("Final Displacement Error: ", fde)
            print("Frechet Distance      : ", df)
            print("Area between two curves: ", area)
            print("Dynamic Time Warping  : ", dtw)
            print("Final Angle Distance  : ", final_angle_distance[0])
            return df, area, dtw, fde, final_angle_distance[0]
            
    def generalize(self, A, b, start, final_distance_label=None, final_distance_angle=None, ax=None):

        _mod1 = self.model.marginal_model(slice(0, 4)).lintrans(A[0], b[0])
        _mod2 = self.model.marginal_model(slice(4, 8)).lintrans(A[1], b[1])
        
        # product 
        _prod = _mod1 * _mod2
        
        # get the most probable sequence of state for this demonstration
        sq = [int(count // (self.horizon_mean/self.nb_states)) for count in range(self.horizon_mean)]
        lqr = PoGLQR(nb_dim=2, dt=0.05, horizon=self.horizon_mean)       
        #sq = [i // (n/m) for i in range(n)]
        # solving LQR with Product of Gaussian, see notebook on LQR
        

        lqr.mvn_xi = _prod.concatenate_gaussian(sq) # augmented version of gaussian
        lqr.mvn_u = -4.
        lqr.x0 = start
        
        xi = lqr.seq_xi
        
        # pbd.plot_gmm(_mod1.mu, _mod1.sigma, swap=True, ax=ax[index_in_training_set], dim=[0, 1], color='steelblue', alpha=0.3)
        # pbd.plot_gmm(_mod2.mu, _mod2.sigma, swap=True, ax=ax[index_in_training_set], dim=[0, 1], color='orangered', alpha=0.3)
        if ax is not None:
            # ax.grid(color='gray', linestyle='-', linewidth=1)
            # Customize the background color
            # ax.set_facecolor('white')
            # pbd.plot_gmm(_prod.mu, _prod.sigma, swap=True, ax=ax, dim=[0, 1], color=[255.0/256.0,140.0/256.0,0.0], alpha=0.5)
            printing_points=np.zeros((4,2))
            print(b[0])
            printing_points[0,:]=b[0][:2]
            printing_points[1,:]=b[0][:2]+A[0][:2,:2] @ np.array([ 0, 10])
            printing_points[2,:]=b[1][:2]
            printing_points[3,:]=b[1][:2]+A[1][:2,:2] @ np.array([ 0, -10])
            ax.plot(xi[:, 0], xi[:, 1], color=[255.0/256.0,20.0/256.0,147.0/256.0], lw=2)
            ax.plot(printing_points[0:2,0],printing_points[0:2,1], linewidth=10, alpha=0.9, c='green')
            ax.plot(printing_points[2:4,0],printing_points[2:4,1], linewidth=10, alpha=0.9, c=[30.0/256.0,144.0/256.0,255.0/256.0])
            ax.scatter(printing_points[0,0],printing_points[0,1], linewidth=10, alpha=0.9, c='green')
            ax.scatter(printing_points[2,0],printing_points[2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])

        if final_distance_label is not None and final_distance_angle is not None: 
            fd=  np.linalg.inv(A[1][0:2,0:2]) @ (xi[-1, 0:2] - b[1][0:2])
            fde=np.linalg.norm(final_distance_label-fd)

            final_vel=  np.linalg.inv(A[1][0:2,0:2]) @ (xi[-1, 0:2] - xi[-5, 0:2])

            final_angle= np.arctan2(final_vel[1], final_vel[0])

            final_angle_distance= np.abs(final_angle - final_distance_angle)

            print("Final Point Distance  : ", fde)
            print("Final Angle Distance  : ", final_angle_distance[0])

            return fde, final_angle_distance[0]