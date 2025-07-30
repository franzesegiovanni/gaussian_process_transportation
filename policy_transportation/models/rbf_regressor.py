import numpy as np
import numbers
from warnings import warn

def gaussian_kernel(x, beta, X=None):
    if X is None:
        X = x
    diff = x[:, None, :] - X[None, :, :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))
def gaussian_kernel_derivative(x, beta, X=None):
    if X is None:
        X = x
    
    X_T= (X).transpose()
    x_T = x.transpose()
    X_reshaped = X_T[:,  np.newaxis,:]
    x_reshaped = x_T[:,  :, np.newaxis]
    # Calculate the difference
    difference_matrix =  X_reshaped - x_reshaped
    
    coefficient= difference_matrix/ ( beta** 2) 
    diff = x[:, None, :] - X[None, :, :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    df_dx =coefficient * np.exp(-diff / (2 * beta**2))
    df_dx= df_dx.transpose(1,0,2)
    return df_dx
class RBFRegression():
    """
    Radial Basis Function regression.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.
    
    low_rank: bool
        Whether to use low rank approximation.
    
    num_eig: int
        Number of eigenvectors to use in lowrank calculation.

    """

    def __init__(self, sigma2=None, max_iterations=None, tolerance=None, w=None, beta=None):
        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kernel. Instead got: {}".format(beta))

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or sigma2 <= 0):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))

        if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
        elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
            warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
            max_iterations = int(max_iterations)

        if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(tolerance))

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))


        self.sigma2 = sigma2
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.w = w
        self.beta = beta
        self.is_residual = False

    def fit(self,X, Y):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        
        # A = np.dot(np.diag(self.P1), self.covar_mat) + \
        #     self.alpha * self.sigma2 * np.eye(self.M)
        # B = self.PX - np.dot(np.diag(self.P1), self.source_distribution)
        # self.W = np.linalg.solve(A, B)

        self.X = X

        (self.N, self.D) = Y.shape
        (self.M, _) = X.shape

        self.P = np.eye(self.M)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, Y)

        self.W = np.zeros((self.M, self.D))
        self.covar_mat = gaussian_kernel(self.X, self.beta)
        A = self.covar_mat + self.sigma2 * np.eye(self.M)
        B = Y -  self.X
        self.W = np.linalg.solve(A, B)


    def predict(self, x, return_std=False):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        Attributes
        ----------
        source_distribution: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.source_distribution used.
        
        Returns
        -------
        If source_distribution is None, returns None.
        Otherwise, returns the transformed source_distribution.
                

        """
        G = gaussian_kernel(x=x, beta=self.beta, X=self.X)
        if return_std:
            std = np.zeros_like(x)
            return x + np.dot(G, self.W), std
        else:
            return x + np.dot(G, self.W)

    def derivative(self, x, return_var=False):
        """
        Calculate the derivative of the transformation at a point.

        Parameters
        ----------
        x: numpy array
            Point at which to calculate the derivative.

        Returns
        -------
        Derivative of the transformation at point x.
        """
        G = gaussian_kernel_derivative(x=x, beta=self.beta, X=self.X)
        GW= np.dot(G, self.W)
        GW= GW.transpose(0,2,1)
        if return_var:
            var = np.zeros_like(GW)
            np.eye(GW.shape[1])[None, :, :] + GW, var
        else:
            return  np.eye(GW.shape[1])[None, :, :] + GW
