import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from policy_transportation import GaussianProcessTransportation as Transport
from generate_random_frame_orientation import generate_frame_orientation
from policy_transportation.plot_utils import draw_error_band
import warnings
import os
import similaritymeasures
import random
warnings.filterwarnings("ignore")
class Multiple_Reference_Frames_GPT:
    def __init__(self):
        pass


    def load_data(self):
        self.distribution=None

    def train(self):
        self.transport=Transport()
        k_transport = C(constant_value=np.sqrt(10))  * Matern(20*np.ones(1), [10,50], nu=2.5) + WhiteKernel(0.01 , [0.0000001, 0.000001])
        self.transport.kernel_transport=k_transport
    def reproduce(self, index_source, ax=None, compute_metrics=False, plot=False):
        X=self.demos_x[index_source]

        self.transport.source_distribution=distribution[index_source,:,:]
        self.transport.target_distribution=distribution_input[index_target,:,:]
        self.transport.training_traj=X
        self.transport.fit_transportation()
        self.transport.apply_transportation()
        X1=self.transport.training_traj
        std=self.transport.std

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

            # fde=np.linalg.norm(demos_x[i][-1]-X1[-1])
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

    def generalize():
        pass