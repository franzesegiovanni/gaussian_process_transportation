import numpy as np
from matplotlib import pyplot as plt
from tp_gmm.TPGMM_GMR import TPGMM_GMR #remember to pip install the python package at https://github.com/franzesegiovanni/tp_gmm 
from tp_gmm.sClass import s
from tp_gmm.pClass import p
from policy_transportation.utils import resample
import os
import similaritymeasures
np.set_printoptions(precision=2) 
class Multiple_reference_frames_TPGMM():
    def __init__(self, nbVar = 3, nbFrames = 2, nbStates = 3, nbData = 20, total_time=1):
        self.nbVar = nbVar
        self.nbFrames = nbFrames
        self.nbStates = nbStates
        self.nbData = nbData
        self.resample_lenght = nbData
        self.total_time=total_time
    def load_data(self, filename):
        # demos = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]
        ### Load data
        demos = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]
        ### Trajectory data
        demos_x = demos['x'] # position
        for i in range(len(demos_x)):
            demos_x[i] = resample(demos_x[i], self.resample_lenght)
            demos_x[i] = np.hstack([np.linspace(0, self.total_time,self.resample_lenght).reshape(-1,1), demos_x[i]]) # 

        demos_A = demos['A']
        demos_b = demos['b']
        self.slist = []
        self.starting_point_rel=[]
        for i in range(len(demos_A)):
            pmat = np.empty(shape=(2, self.resample_lenght), dtype=object)
            tempData = demos_x[i].transpose()
            for j in range(2):
                for k in range(self.resample_lenght):
                    A=np.eye(3)
                    B=np.zeros((3,1))    
                    A[1:,1:]=demos_A[i][0][j]
                    B[1:,0]=demos_b[i][0][j].reshape(-1,)
                    A[1:,1:]=-np.eye(2) @ A[1:,1:]
                    pmat[j, k] = p(A, B, np.linalg.inv(A), 3)
            self.starting_point_rel.append(tempData[1:,0]- demos_b[i][0][0].reshape(-1,))
            self.slist.append(s(pmat, tempData, tempData.shape[1], 3))

        self.demos_x=demos_x  

        self.print_points_frames=np.zeros((len(demos_x),4,2))
        # print_points_frames_new=np.zeros((len(demos_x),4,2))
        # final_distance=np.zeros((len(demos_x),2))
        # final_orientation=np.zeros((len(demos_x),1))
        for i in range(len(demos_x)):
            self.print_points_frames[i,0,:]=demos_b[i][0][0]
            self.print_points_frames[i,1,:]=demos_b[i][0][0]+demos_A[i][0][0] @ np.array([ 0, 10])
            self.print_points_frames[i,2,:]=demos_b[i][0][1]
            self.print_points_frames[i,3,:]=demos_b[i][0][1]+demos_A[i][0][1] @ np.array([ 0, -10])

        self.demos_A=demos_A
        self.demos_b=demos_b

        self.final_distance=np.zeros((len(demos_x),2))
        self.final_orientation=np.zeros((len(demos_x),1))


        for i in range(len(demos_x)):
            self.final_distance[i]=  np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,1:] - demos_b[i][0][1])
            final_delta=np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,1:]-demos_x[i][-5,1:])
            self.final_orientation[i]= np.arctan2(final_delta[1],final_delta[0])


    def train(self, index_partial_dataset=None):
        # This code allows to train the model on a subset of the original training set, it is enough to specify the indexes of the demos to be used in the input list. If nothing is specified, the model is trained on the whole training set
        if index_partial_dataset is None:
            index_partial_dataset=list(range(len(self.demos_x)))
        self.policy = TPGMM_GMR(self.nbStates, self.nbFrames, self.nbVar)

        slist_partial=[self.slist[index] for index in index_partial_dataset]
        # Learning the model
        self.policy.fit(slist_partial)

    def reproduce(self, ax=None, compute_metrics=False, index_partial_dataset=None):
        if index_partial_dataset is None:
            index_partial_dataset=list(range(len(self.demos_x)))
        rnewlist= [None] * len(self.demos_x)
        for i in index_partial_dataset: 
            pmat_new = np.empty(shape=(2, self.nbData), dtype=object)
            for j in range(2):
                for k in range(self.nbData):
                    A=np.eye(3)
                    B=np.zeros((3,1))
                    A[1:,1:]=self.demos_A[i][0][j]
                    B[1:,0]=self.demos_b[i][0][j].reshape(-1,)
                    A[1:,1:]=-np.eye(2) @ A[1:,1:]
                    pmat_new[j, k] = p(A, B, np.linalg.inv(A), 3)
            starting_point_rel_new=self.starting_point_rel[i]+ self.demos_b[i][0][0].reshape(-1,)
            rnewlist[i]=self.policy.reproduce(pmat_new, starting_point_rel_new, dt=self.total_time/self.resample_lenght)
        if ax is not None:
            #plot the new trajectories
            for i in index_partial_dataset:
                ax.plot(self.print_points_frames[i][0:2,0],self.print_points_frames[i][0:2,1], linewidth=10, alpha=0.9, c='green')
                ax.plot(self.print_points_frames[i][2:4,0],self.print_points_frames[i][2:4,1], linewidth=10, alpha=0.9, c=[30.0/256.0,144.0/256.0,255.0/256.0])
                ax.scatter(self.print_points_frames[i][0,0],self.print_points_frames[i][0,1], linewidth=10, alpha=0.9, c='green')
                ax.scatter(self.print_points_frames[i][2,0],self.print_points_frames[i][2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
                ax.plot(rnewlist[i].Data[1,:], rnewlist[i].Data[2,:], color=[255.0/256.0,20.0/256.0,147.0/256.0])
                # ax.plot(self.demos_x[i][:,1].transpose(), self.demos_x[i][:,2].transpose(), 'k--', lw=2)
                ax.scatter(self.demos_x[i][:,1].transpose(), self.demos_x[i][:,2].transpose())

        if compute_metrics:
            tot_df=[]
            tot_area=[]
            tot_dtw=[]
            tot_fde=[]
            tot_final_angle_distance=[]
            for i in index_partial_dataset:

                df = similaritymeasures.frechet_dist(self.demos_x[i][:,1:].transpose(), rnewlist[i].Data[1:,:])
                # area between two curves
                area = similaritymeasures.area_between_two_curves(self.demos_x[i][:,1:].transpose(), rnewlist[i].Data[1:,:])
                # Dynamic Time Warping distance
                dtw, d = similaritymeasures.dtw(self.demos_x[i][:,1:].transpose(), rnewlist[i].Data[1:,:])
                # Final Displacement Error
                # fde = np.linalg.norm(demos_x[index_in_training_set][-1] - xi[-1,0:2])
                fd=  np.linalg.inv(self.demos_A[i][0][1]) @ (rnewlist[i].Data[1:,-1] - self.demos_b[i][0][1])
                fde=np.linalg.norm(self.final_distance[i]-fd)

                final_vel=  np.linalg.inv(self.demos_A[i][0][1]) @ (rnewlist[i].Data[1:,-1] - rnewlist[i].Data[1:,-5])

                final_angle= np.arctan2(final_vel[1], final_vel[0])

                final_angle_distance= np.abs(final_angle - self.final_orientation[i])

                print("Final Displacement Error: ", fde)
                print("Frechet Distance      : ", df)
                print("Area between two curves: ", area)
                print("Dynamic Time Warping  : ", dtw)
                print("Final Angle Distance  : ", final_angle_distance[0])
                tot_df.append(df)
                tot_area.append(area)
                tot_dtw.append(dtw)
                tot_fde.append(fde)
                tot_final_angle_distance.append(final_angle_distance[0])

            return tot_df, tot_area, tot_dtw, tot_fde, tot_final_angle_distance
        
    def generalize(self, input_A, input_b, start, ax=None, final_distance_label=None, final_angle_label=None):
        print_points_frames_new=np.zeros((4,2))
        print_points_frames_new[0,:]=input_b[0]
        print_points_frames_new[1,:]=input_b[0]+input_A[0] @ np.array([ 0, 10])
        print_points_frames_new[2,:]=input_b[1]
        print_points_frames_new[3,:]=input_b[1]+input_A[1] @ np.array([ 0, -10])

        rnewlist=[]

        pmat_new = np.empty(shape=(2, self.resample_lenght), dtype=object)
        for j in range(2):
            for k in range(self.resample_lenght):
                A=np.eye(3)
                B=np.zeros((3,1))
                A[1:,1:]=input_A[j]
                B[1:,0]=input_b[j].reshape(-1,)
                A[1:,1:]=-np.eye(2) @ A[1:,1:]
                pmat_new[j, k] = p(A, B, np.linalg.inv(A), 3)
        # starting_point_rel_new=starting_point_rel+ demos_b_new[i][0][0].reshape(-1,)
        # print(starting_point_rel_new)       
        # rnewlist.append(self.policy.reproduce(pmat_new, start))
        rnew=self.policy.reproduce(pmat_new, start, dt=self.total_time/self.resample_lenght)

        if ax is not None:
            # for i in range(len(A)):
            # plt.figure()
            ax.plot(print_points_frames_new[0:2,0],print_points_frames_new[0:2,1], linewidth=10, alpha=0.9, c='green')
            ax.plot(print_points_frames_new[2:4,0],print_points_frames_new[2:4,1], linewidth=10, alpha=0.9, c=[30.0/256.0,144.0/256.0,255.0/256.0])
            ax.scatter(print_points_frames_new[0,0],print_points_frames_new[0,1], linewidth=10, alpha=0.9, c='green')
            ax.scatter(print_points_frames_new[2,0],print_points_frames_new[2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
            ax.plot(rnew.Data[1,:], rnew.Data[2,:], color=[255.0/256.0,20.0/256.0,147.0/256.0])

        if final_distance_label is not None and final_angle_label is not None: 
            fd=  np.linalg.inv(input_A[1]) @ (rnew.Data[1:,-1] - input_b[1])
            fde=np.linalg.norm(fd-final_distance_label)

            final_vel=  np.linalg.inv(input_A[1]) @ (rnew.Data[1:,-1] - rnew.Data[1:,-5])

            final_angle= np.arctan2(final_vel[1], final_vel[0])

            final_angle_distance= np.abs(final_angle - final_angle_label)

            print("Final Point Distance  : ", fde)
            print("Final Angle Distance  : ", final_angle_distance[0])

            return fde, final_angle_distance[0]    




