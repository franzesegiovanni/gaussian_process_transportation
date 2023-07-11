from gmr.sklearn import GaussianMixtureRegressor as GMM
import numpy as np
import matplotlib.pyplot as plt
X=np.linspace(0, 1, 100).reshape(-1,1)
Y=np.sin(2*np.pi*X)
gmm = GMM(n_components=10, random_state=0)
gmm.fit(X, Y)
X1=np.linspace(0, 1, 100).reshape(-1,1)
Y1=gmm.predict(X1)
plt.plot(X1,Y1)
plt.scatter(X,Y)
plt.show()