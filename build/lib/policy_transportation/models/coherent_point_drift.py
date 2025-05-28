
from builtins import super
import numpy as np
import numbers
from warnings import warn

def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))

class DeformableRegistration():
    """
    Deformable registration.

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

    def __init__(self, target_distribution, source_distribution, sigma2=None, max_iterations=None, tolerance=None, w=None, beta=None):
        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(beta))
        
        if type(target_distribution) is not np.ndarray or target_distribution.ndim != 2:
            raise ValueError(
                "The target point cloud (target_distribution) must be at a 2D numpy array.")

        if type(source_distribution) is not np.ndarray or source_distribution.ndim != 2:
            raise ValueError(
                "The source point cloud (source_distribution) must be a 2D numpy array.")

        if target_distribution.shape[1] != source_distribution.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions.")

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

        self.target_distribution = target_distribution
        self.source_distribution = source_distribution
        self.deformed_source_distribution = source_distribution
        self.sigma2 = sigma2
        (self.N, self.D) = self.target_distribution.shape
        (self.M, _) = self.source_distribution.shape

        self.P = np.eye(self.M)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.target_distribution)

        self.beta = beta
        self.W = np.zeros((self.M, self.D))
        self.covar_mat = gaussian_kernel(self.source_distribution, self.beta)

    def fit(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        
        # A = np.dot(np.diag(self.P1), self.covar_mat) + \
        #     self.alpha * self.sigma2 * np.eye(self.M)
        # B = self.PX - np.dot(np.diag(self.P1), self.source_distribution)
        # self.W = np.linalg.solve(A, B)

        A = self.covar_mat + self.sigma2 * np.eye(self.M)
        B = self.target_distribution -  self.source_distribution
        self.W = np.linalg.solve(A, B)


    def predict(self, x):
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
        G = gaussian_kernel(X=x, beta=self.beta, Y=self.source_distribution)
        return x + np.dot(G, self.W)
