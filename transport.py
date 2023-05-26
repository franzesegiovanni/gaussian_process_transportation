from gaussian_process import GaussianProcess
import pickle
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import quaternion
class Transport():
    def __init__(self):
        super(Transport, self).__init__()

    # def record_source_distribution(self):
    #     self.source_distribution=self.detections

    # def record_target_distribution(self):
    #     self.target_distribution=self.detections    

    def save_distributions(self):
        # create a binary pickle file 
        f = open("distributions/source.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(self.source_distribution,f)
        # close file
        f.close()

    # create a binary pickle file 
        f = open("distributions/target.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(self.target_distribution,f)
        # close file
        f.close()

    def load_distributions(self):
        try:
            with open("distributions/source.pkl","rb") as source:
                self.source_distribution = pickle.load(source)
        except:
            print("No source distribution saved")

        try:
            with open("distributions/target.pkl","rb") as target:
                self.target_distribution = pickle.load(target)
        except:
            print("No target distribution saved")    



    def Policy_Transport(self, verboose=False):
        self.convert_distribution_to_array()
        delta_map = self.target_distribution - self.source_distribution
        kernel = C(0.1,[0.1,0.1]) * RBF(length_scale=[0.5], length_scale_bounds=[0.3,1.0]) + WhiteKernel(0.0001, [0.0001,0.0001]) # test based on sim

        self.gp_delta_map=GaussianProcess(kernel=kernel, n_restarts_optimizer=20)
        self.gp_delta_map.fit(self.source_distribution, delta_map)        

        #Deform Trajactories 
        delta_map_mean, _= self.gp_delta_map.predict(self.training_traj)
        transported_traj = self.training_traj + delta_map_mean 

        #Deform Deltas and orientation
        new_delta = np.ones((len(self.training_traj),3))
        for i in range(len(self.training_traj[:,0])):
            pos=(np.array(self.training_traj[i,:]).reshape(1,-1))
            [Jacobian,_]=self.gp_delta_map.derivative(pos)
            new_delta[i]=(self.training_delta[i]+np.matmul(np.transpose(Jacobian[0]),self.training_delta[i]))
            rot=np.eye(3) + np.transpose(Jacobian[0])
            rot_norm=rot/np.linalg.det(rot)
            quat_deformation=quaternion.from_rotation_matrix(rot_norm, nonorthogonal=True)
            curr_quat=quaternion.from_float_array(self.training_ori[i,:])
            product_quat=quat_deformation*curr_quat
            self.training_ori[i,:]=np.array([product_quat.w, product_quat.x, product_quat.y, product_quat.z])

        #Update the trajectory and the delta     
        self.training_traj=transported_traj
        self.training_delta=new_delta