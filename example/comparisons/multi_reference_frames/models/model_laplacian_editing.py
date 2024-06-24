import numpy as np
from matplotlib import pyplot as plt
from policy_transportation.transportation.laplacian_editing_transportation import LaplacianEditingTransportation as Transport
from generate_random_frame_orientation import generate_frame_orientation
from policy_transportation.plot_utils import draw_error_band
import warnings
import os
import similaritymeasures
import random
warnings.filterwarnings("ignore")
class Multiple_Reference_Frames_LA:
    def __init__(self):
        self.transport=Transport()

    def generate_distribution_from_frames(self, A,b):
        distribution_training_set=np.zeros((len(A), 4,2))
        frame_dim=5
        for i in range(len(A)):
            distribution_training_set[i,0,:]=b[i][0][0]
            distribution_training_set[i,1,:]=b[i][0][0]+A[i][0][0] @ np.array([ 0, frame_dim])
            distribution_training_set[i,2,:]=b[i][0][1]
            distribution_training_set[i,3,:]=b[i][0][1]+A[i][0][1] @ np.array([ 0, -frame_dim])
        return distribution_training_set   
    
    def load_dataset(self, filename = 'reach_target'):
        

        demos = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]

        ### Trajectory data
        demos_x = demos['x'] # position

        ### Coordinate systems transformation
        demos_A = [d for d in demos['A']]
        demos_b = [d for d in demos['b']]

        distribution_training_set=np.zeros((len(demos_x),10,2))
        final_distance=np.zeros((len(demos_x),2))
        final_orientation=np.zeros((len(demos_x),1))
        # index=2
        distribution_training_set=self.generate_distribution_from_frames(demos_A,demos_b)
        for i in range(len(demos_x)):
            final_distance[i]=  np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,:] - demos_b[i][0][1])

            final_delta=np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,:]-demos_x[i][-2,:])
            final_orientation[i]= np.arctan2(final_delta[1],final_delta[0])

        self.demos_x=demos_x
        self.demos_A=demos_A
        self.demos_b=demos_b
        self.distribution_training_set=distribution_training_set
        self.final_distance=final_distance
        self.final_orientation=final_orientation    

    def load_test_dataset(self, test_A, test_b):

        distribution_test_set=np.zeros((len(test_A),10,2))

        distribution_test_set=self.generate_distribution_from_frames(test_A,test_b)     
        self.distribution_test_set=distribution_test_set
        self.test_A=test_A
        self.test_b=test_b

    def reproduce(self, index_source, index_target, ax=None, compute_metrics=False):
        X=self.demos_x[index_source]

        self.transport.source_distribution=self.distribution_training_set[index_source,:,:]
        self.transport.target_distribution=self.distribution_training_set[index_target,:,:]
        self.transport.training_traj=X

        
        self.transport.fit_transportation()
        self.transport.apply_transportation()
        std=self.transport.std

        X1=self.transport.training_traj
        

        if ax is not None:
            self.plot(X1, std, self.distribution_training_set[index_target,:,:], ax)
            ax.plot(self.demos_x[index_target][:,0],self.demos_x[index_target][:,1], 'k--')

        if compute_metrics==True:    
        # Discrete Frechet distance
            df = similaritymeasures.frechet_dist(self.demos_x[index_target], X1)

            # quantify the difference between the two curves using
            # area between two curves
            area = similaritymeasures.area_between_two_curves(self.demos_x[index_target], X1)

            # quantify the difference between the two curves using
            # Dynamic Time Warping distance
            dtw, d = similaritymeasures.dtw(self.demos_x[index_target], X1)

            # fde=np.linalg.norm(demos_x[i][-1]-X1[-1])
            fd=  np.linalg.inv(self.demos_A[index_target][0][1]) @ (X1[-1] - self.demos_b[index_target][0][1])
            fde=np.linalg.norm(self.final_distance[index_target]-fd)

            final_vel=  np.linalg.inv(self.demos_A[index_target][0][1]) @ (X1[-1] - X1[-5])

            final_angle= np.arctan2(final_vel[1], final_vel[0])

            final_angle_distance= np.abs(final_angle - self.final_orientation[index_target])  

            print("Final Point Distance  : ", fde)
            print("Frechet Distance      : ", df)
            print("Area between two curves: ", area)
            print("Dynamic Time Warping  : ", dtw)
            print("Final Angle Distance  : ", final_angle_distance[0])
            return df, area, dtw, fde, final_angle_distance[0]

    def generalize(self, index_source, index_target, ax=None, compute_metrics=False, linear=False):
        X=self.demos_x[index_source].reshape(-1,2)

        self.transport.source_distribution=self.distribution_training_set[index_source,:,:].reshape(-1,2)
        self.transport.target_distribution=self.distribution_test_set[index_target,:,:].reshape(-1,2)
        self.transport.training_traj=X

        self.transport.fit_transportation()
        self.transport.apply_transportation()
        std=self.transport.std
        X1=self.transport.training_traj
        if ax is not None:
            self.plot(X1, std, self.distribution_test_set[index_target,:,:], ax=ax)


        if compute_metrics==True:    
            # fde=np.linalg.norm(transport.target_distribution[2,:]-X1[-1])
            fd=  np.linalg.inv(self.test_A[index_target][0][1]) @ (X1[-1] - self.test_b[index_target][0][1])
            fde=np.linalg.norm(self.final_distance[index_target]-fd)

            final_vel=  np.linalg.inv(self.test_A[index_target][0][1]) @ (X1[-1] - X1[-5])

            final_angle= np.arctan2(final_vel[1], final_vel[0])

            final_angle_distance= np.abs(final_angle - self.final_orientation[index_target])

            print("Final Point Distance  : ", fde)
            print("Final Angle Distance  : ", final_angle_distance[0])
            return fde, final_angle_distance[0]
    
    def plot(self, X1, std, distribution, ax=None):
        ax.plot(distribution[0:2,0],distribution[0:2,1], linewidth=10, alpha=0.9, c='green')
        ax.scatter(distribution[0,0],distribution[0,1], linewidth=10, alpha=0.9, c='green')
        ax.plot(distribution[2:4,0],distribution[2:4,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
        ax.scatter(distribution[2,0],distribution[2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
        # ax.plot(distribution[:,0],distribution[:,1], 'b*',  linewidth=0.2)
        ax.plot(X1[:,0],X1[:,1], c= [255.0/256.0,20.0/256.0,147.0/256.0])