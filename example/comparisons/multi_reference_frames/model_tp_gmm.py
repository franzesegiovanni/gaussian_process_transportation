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
    def __init__(self, nbVar = 3, nbFrames = 2, nbStates = 3, nbData = 100):
        self.nbVar = nbVar
        self.nbFrames = nbFrames
        self.nbStates = nbStates
        self.nbData = nbData
        self.resample_lenght = nbData
    def load_data(self, filename):
        # demos = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]
        ### Load data
        pbd_path = os. getcwd() 
        filename = '/data/reach_target'
        demos = np.load(pbd_path + filename + '.npy', allow_pickle=True, encoding='latin1')[()]
        ### Trajectory data
        demos_x = demos['x'] # position
        for i in range(len(demos_x)):
            demos_x[i] = resample(demos_x[i], self.resample_lenght)
            demos_x[i] = np.hstack([np.linspace(0, 2,self.resample_lenght).reshape(-1,1), demos_x[i]]) # 

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
    def train(self):
        self.policy = TPGMM_GMR(self.nbStates, self.nbFrames, self.nbVar)

        # Learning the model
        self.policy.fit(self.slist)

    def reproduce(self, plot=False, index_in_training_set=None, compute_metrics=False):
        rnewlist=[]
        if index_in_training_set is None:
            index_in_training_set=range(len(self.demos_A))
        for i in index_in_training_set: 
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
            rnewlist.append(self.policy.reproduce(pmat_new, starting_point_rel_new))

        if plot:
            #plot the new trajectories
            plt.figure()
            for i in range(len(self.demos_A)):
                plt.plot(self.print_points_frames[i][0:2,0],self.print_points_frames[i][0:2,1], linewidth=10, alpha=0.9, c='green')
                plt.plot(self.print_points_frames[i][2:4,0],self.print_points_frames[i][2:4,1], linewidth=10, alpha=0.9, c=[30.0/256.0,144.0/256.0,255.0/256.0])
                plt.scatter(self.print_points_frames[i][0,0],self.print_points_frames[i][0,1], linewidth=10, alpha=0.9, c='green')
                plt.scatter(self.print_points_frames[i][2,0],self.print_points_frames[i][2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
                plt.plot(rnewlist[i].Data[1,:], rnewlist[i].Data[2,:], 'b')

        if compute_metrics:
            tot_df=[]
            tot_area=[]
            tot_dtw=[]
            tot_fde=[]
            tot_final_angle_distance=[]
            
            for i in index_in_training_set:
                df = similaritymeasures.frechet_dist(self.demos_x[i], rnewlist[i].Data[1:,:])
                # area between two curves
                area = similaritymeasures.area_between_two_curves(self.demos_x[i], rnewlist[i].Data[1:,:])
                # Dynamic Time Warping distance
                dtw, d = similaritymeasures.dtw(self.demos_x[i], rnewlist[i].Data[1:,:])
                # Final Displacement Error
                # fde = np.linalg.norm(demos_x[index_in_training_set][-1] - xi[-1,0:2])
                fd=  np.linalg.inv(A[1][0:2,0:2]) @ (rnewlist[i].Data[1:,-1] - B[1][0:2])
                fde=np.linalg.norm(self.final_distance[i]-fd)

                final_vel=  np.linalg.inv(A[1][0:2,0:2]) @ (rnewlist[i].Data[1:,-1] - rnewlist[i].Data[1:,-5])

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
            return df, area, dtw, fde, final_angle_distance[0]
            
    def generalize(self, input_A, input_b, start, ax=None, plot=True, compute_metrics=False):
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
        rnew=self.policy.reproduce(pmat_new, start)

        if plot:
            # for i in range(len(A)):
            # plt.figure()
            ax.plot(print_points_frames_new[0:2,0],print_points_frames_new[0:2,1], linewidth=10, alpha=0.9, c='green')
            ax.plot(print_points_frames_new[2:4,0],print_points_frames_new[2:4,1], linewidth=10, alpha=0.9, c=[30.0/256.0,144.0/256.0,255.0/256.0])
            ax.scatter(print_points_frames_new[0,0],print_points_frames_new[0,1], linewidth=10, alpha=0.9, c='green')
            ax.scatter(print_points_frames_new[2,0],print_points_frames_new[2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
            ax.plot(rnew.Data[1,:], rnew.Data[2,:], 'b')

        if compute_metrics:
            pass


