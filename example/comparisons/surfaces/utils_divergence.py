import numpy as np
import scipy
# https://scoste.fr/posts/dkl_gaussian/#:~:text=k%20l%20(%20P%20%E2%88%A5%20Q,%E2%88%92%20p%20(%20x%20)%20ln%20%E2%81%A1

# https://gist.github.com/ChuaCheowHuan/18977a3e77c0655d945e8af60633e4df

def kl_mvn(to, fr):
    """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices"""
    m_to, S_to = to
    m_fr, S_fr = fr
    
    d = m_fr - m_to
    
    c, lower = scipy.linalg.cho_factor(S_fr)
    def solve(B):
        return scipy.linalg.cho_solve((c, lower), B)
    
    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    return (term1 + term2 + term3 - len(d))/2.

def compute_distance(traj1, traj2, std1, std2):
    # Check that the vectors have the same length
    if len(traj1) != len(traj2):
        raise ValueError("Vectors must have the same length.")

    # Check that the variances have the same length as the vectors
    if len(traj1) != len(traj1) or len(traj2) != len(traj2):
        raise ValueError("Variances must have the same length as the vectors.")

    # Calculate the weighted Euclidean distance for each element in the vectors
    distances = []
    for i in range(len(traj1)):
        distance = np.sqrt( (traj1[i] - traj2[i]) **2  / (std1[i] **2) +  (traj1[i] - traj2[i]) ** 2 / (std2[i]**2) )
        # print(distance.shape)   
        distance=np.sqrt(np.sum(distance**2))
        distances.append(distance)

    # Compute the overall similarity measure by aggregating the weighted distances
    similarity = np.mean(distances)  # Use np.sum() if you prefer a sum-based similarity measure

    return similarity


def compute_distance_euclidean(traj1, traj2):
    # Check that the vectors have the same length
    if len(traj1) != len(traj2):
        raise ValueError("Vectors must have the same length.")

    # Check that the variances have the same length as the vectors
    if len(traj1) != len(traj1) or len(traj2) != len(traj2):
        raise ValueError("Variances must have the same length as the vectors.")

    # Calculate the weighted Euclidean distance for each element in the vectors
    distances = []
    for i in range(len(traj1)):
        distance = np.sqrt( (traj1[i] - traj2[i]) **2  + (traj1[i] - traj2[i]) ** 2 )
        # print(distance.shape)   
        distance=np.sqrt(np.sum(distance**2))
        distances.append(distance)

    # Compute the overall similarity measure by aggregating the weighted distances
    similarity = np.mean(distances)  # Use np.sum() if you prefer a sum-based similarity measure

    return similarity