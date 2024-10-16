"""
Authors: Giovanni Franzese & Ravi Prakash, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import optuna
from policy_transportation import GaussianProcess, AffineTransform
import pickle
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import quaternion
import matplotlib.pyplot as plt
class GaussianProcessTransportationDiffeo():
    def __init__(self, kernel_transport=C(0.1) * RBF(length_scale=[0.1]) + WhiteKernel(0.0001)):
        super(GaussianProcessTransportationDiffeo, self).__init__()
        self.kernel_transport=kernel_transport

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


    def fit_transportation(self, optimize=True, do_scale=False, do_rotation=True):
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        self.delta_distribution = self.target_distribution - source_distribution

            
        print("Kernel:", self.kernel_transport)
        if optimize==True:    
            self.gp_delta_map=GaussianProcess(kernel=self.kernel_transport, n_restarts_optimizer=5)
        else:
            self.gp_delta_map=GaussianProcess(kernel=self.kernel_transport, optimizer=None)    
        self.gp_delta_map.fit(source_distribution, self.delta_distribution)  
        self.kernel_transport=self.gp_delta_map.kernel


    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.delta_map_mean, self.std= self.gp_delta_map.predict(self.traj_rotated, return_std=True)
        self.training_traj = self.traj_rotated + self.delta_map_mean 

        #Deform Deltas and orientation
        if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
            pos=(np.array(self.traj_rotated))
            Jacobian, Jacobain_var=self.gp_delta_map.derivative(pos, return_var=True)
            # J_var = Jacobain_var[:,0,:]

            rot_gp= np.eye(Jacobian[0].shape[0]) + Jacobian
            rot_affine= self.affine_transform.rotation_matrix
            derivative_affine= self.affine_transform.derivative(pos)


        if  hasattr(self, 'training_delta'):
            self.training_delta = self.training_delta[:,:,np.newaxis]

            self.training_delta=  derivative_affine @ self.training_delta
            self.var_vel_transported=Jacobain_var @ self.training_delta**2

            self.training_delta= rot_gp @ self.training_delta
            self.training_delta=self.training_delta[:,:,0]
            self.var_vel_transported=self.var_vel_transported[:,:,0]


        if  hasattr(self, 'training_ori'):   
            quat_demo=quaternion.from_float_array(self.training_ori)
            quat_affine= quaternion.from_rotation_matrix(rot_affine)
            quat_gp = quaternion.from_rotation_matrix(rot_gp, nonorthogonal=True)
            quat_transport=quat_gp * (quat_affine * quat_demo)
            self.training_ori= quaternion.as_float_array(quat_transport)

    def sample_transportation(self):
        delta_map_samples= self.gp_delta_map.samples(self.traj_rotated)
        training_traj_samples = self.traj_rotated + delta_map_samples 
        return training_traj_samples


    def check_invertibility(self):
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.delta_map_mean, self.std= self.gp_delta_map.predict(self.traj_rotated, return_std=True)

        self.training_traj = self.traj_rotated + self.delta_map_mean 
        delta_inv= - self.delta_distribution
        self.gp_delta_inv=GaussianProcess(kernel=self.kernel_transport, optimizer=None)   
        self.gp_delta_inv.fit(self.target_distribution, delta_inv)
        self.delta_map_inv_mean=self.gp_delta_inv.predict(self.training_traj)[0]
        self.traj_rotated_inv=self.training_traj+ self.delta_map_inv_mean
        error=np.sum(np.linalg.norm(self.delta_map_mean+self.delta_map_inv_mean, axis=1))
        return error

    def diffeomorphism_error(self, trial):
        max_lengthscale = trial.suggest_float("max_lengthscale", 2, 20, log=True)
        self.kernel_transport=C(0.1) * RBF(length_scale=[2, 2], length_scale_bounds=[0.1,max_lengthscale ]) + WhiteKernel(0.0001)
        self.fit_transportation() 
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.delta_map_mean, self.std= self.gp_delta_map.predict(self.traj_rotated, return_std=True)

        self.training_traj_target = self.traj_rotated + self.delta_map_mean 
        delta_inv= - self.delta_distribution
        self.gp_delta_inv=GaussianProcess(kernel=self.kernel_transport, optimizer=None)   
        self.gp_delta_inv.fit(self.target_distribution, delta_inv)
        self.delta_map_inv_mean=self.gp_delta_inv.predict(self.training_traj_target)[0]
        self.traj_rotated_inv=self.training_traj_target+ self.delta_map_inv_mean
        error=np.sum(np.linalg.norm(self.delta_map_mean+self.delta_map_inv_mean, axis=1))
        return error
    
    def optimize_diffeomorphism(self, n_trials=100):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.diffeomorphism_error, n_trials=n_trials)
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        self.kernel_transport=C(0.1) * RBF(length_scale=np.ones(self.training_traj.shape[1]), length_scale_bounds=[1,trial.params['max_lengthscale'] ]) + WhiteKernel(0.0001)
        self.fit_transportation() 
    
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.delta_map_mean, self.std= self.gp_delta_map.predict(self.traj_rotated, return_std=True)

        self.training_traj_target = self.traj_rotated + self.delta_map_mean 
        delta_inv= - self.delta_distribution
        self.gp_delta_inv=GaussianProcess(kernel=self.kernel_transport, optimizer=None)   
        self.gp_delta_inv.fit(self.target_distribution, delta_inv)
        self.delta_map_inv_mean=self.gp_delta_inv.predict(self.training_traj_target)[0]
        self.traj_rotated_inv=self.training_traj_target+ self.delta_map_inv_mean
        plt.figure()
        plt.scatter(self.traj_rotated_inv[:,0], self.traj_rotated_inv[:,1], label='inverse')
        plt.scatter(self.traj_rotated[:,0], self.traj_rotated[:,1], label='original')
        plt.scatter(self.target_distribution[:,0], self.target_distribution[:,1], label='target')
        plt.scatter(self.training_traj_target[:,0], self.training_traj_target[:,1], label='deformed')
        plt.legend()
        plt.show()
