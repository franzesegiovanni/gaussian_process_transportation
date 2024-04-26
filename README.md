# Generalization of Task Parameterized Dynamical Systems using Gaussian Process Transportation

You can read the pre-print of the paper here https://arxiv.org/pdf/2404.13458. 

This article proposes a novel task parameterization and generalization to transport the
original robot policy, i.e., position, velocity, orientation, and stiffness. Unlike the state of the art, only a set of points are tracked during the demonstration and the execution, e.g., a point cloud of
the surface to clean. We then propose to fit a non-linear transformation that would deform the space and then the original policy using the paired source and target point sets. The use of function approximators like Gaussian Processes allows us to generalize, or transport, the policy from every space location while estimating the uncertainty of the resulting policy due to the limited points in the task parameterization point set and the reduced number of demonstrations. We compare the algorithm’s performance with state-of-the-art task parameterization alternatives and analyze the effect of different function approximators. We also validated the algorithm on robot manipulation tasks, i.e., different posture arm dressing, different location product reshelving, and different shape surface cleaning. A video of the experiments can be found here: https://youtu.be/FDmWF7K15KU.

## Installation

```
pip install . 
```

In the example folder you can find the code to reproduce the experiments in the paper. 

The transportation function are all in the `transportation` folder.


If you find my research useful or insightful, please consider citing the paper, star the code and send me an email with your feedback and questions. 

```
Franzese, G., Prakash, R. and Kober, J., 2024. Generalization of Task Parameterized Dynamical Systems using Gaussian Process Transportation. arXiv preprint arXiv:2404.13458.
```

