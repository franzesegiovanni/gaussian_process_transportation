# This code is heavily based on the code from https://github.com/BatyaGG/Task-Parameterized-Gaussian-Mixture-Model
#if you want to run it, you must copy this file and A.npy and B.npy in the original repository.

import numpy as np
from matplotlib import pyplot as plt
from TPGMM_GMR import TPGMM_GMR
from copy import deepcopy

class p:
    def __init__(self, A, b, invA, nbStates):
        self.A = A
        self.b = b
        self.invA = invA
        self.Mu = np.zeros(shape=(len(b),nbStates))
        self.Sigma = np.zeros(shape=(len(b),len(b),nbStates))


class s:
    def __init__(self, p, Data, nbData, nbStates):
        self.p = p
        self.Data = Data
        self.nbData = nbData
        self.GAMMA0 = np.zeros(shape=(nbStates, self.nbData))
        self.GAMMA = None        
# Initialization of parameters and properties------------------------------------------------------------------------- #
nbSamples = 4
nbVar = 3
nbFrames = 2
nbStates = 3
nbData = 200

# Preparing the samples----------------------------------------------------------------------------------------------- #
slist = []
for i in range(nbSamples):
    pmat = np.empty(shape=(nbFrames, nbData), dtype=object)
    tempData = np.loadtxt('data/sample' + str(i + 1) + '_Data.txt', delimiter=',')
    for j in range(nbFrames):
        tempA = np.loadtxt('data/sample' + str(i + 1) + '_frame' + str(j + 1) + '_A.txt', delimiter=',')
        tempB = np.loadtxt('data/sample' + str(i + 1) + '_frame' + str(j + 1) + '_b.txt', delimiter=',')
        for k in range(nbData):
            pmat[j, k] = p(tempA[:, 3*k : 3*k + 3], tempB[:, k].reshape(len(tempB[:, k]), 1),
                           np.linalg.inv(tempA[:, 3*k : 3*k + 3]), nbStates)
    slist.append(s(pmat, tempData, tempData.shape[1], nbStates))

# Creating instance of TPGMM_GMR-------------------------------------------------------------------------------------- #
TPGMMGMR = TPGMM_GMR(nbStates, nbFrames, nbVar)

# Learning the model-------------------------------------------------------------------------------------------------- #
TPGMMGMR.fit(slist)

# Reproduction for parameters used in demonstration------------------------------------------------------------------- #
rdemolist = []
for n in range(nbSamples):
    rdemolist.append(TPGMMGMR.reproduce(slist[n].p, slist[n].Data[1:3,0]))

# distribution_new= np.load('distribution_new.npy')
# Plotting------------------------------------------------------------------------------------------------------------ #
xaxis = 1
yaxis = 2
xlim = [-1.2, 0.8]
ylim = [-1.1, 0.9]

# ax2 = fig.add_subplot(132)
ax =plt.subplot()
plt.xlim(-1,1)
plt.ylim(-1,1)
# plt.set_xlim(xlim)
# plt.set_ylim(ylim)
# plt.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
# plt.title('Reproductions with same task parameters')
for n in range(nbSamples):
    TPGMMGMR.plotReproduction(rdemolist[n], xaxis, yaxis, ax, showGaussians=True)
plt.grid('on')
# plt.show()



plt.figure()
# Reproduction with generated parameters------------------------------------------------------------------------------ #
rnewlist = []
A= np.load('A.npy')
B= np.load('B.npy')
# Reproduction for parameters used in demonstration------------------------------------------------------------------- #

for n in range(nbSamples):
    newP = deepcopy(slist[n].p)
    for m in range(0, nbFrames):
        for k in range(nbData):
            # newP[m, k].A = np.zeros((3,3))
            # newP[m, k].b = np.zeros((3,1))           
            newP[m, k].A = A[n,m,:,:]
            newP[m, k].b= B[n, m ,:].reshape(-1,1)
    rnewlist.append(TPGMMGMR.reproduce(newP, B[n, 0 ,1:]))

# Reproduction with generated parameters------------------------------------------------------------------------------ #
# rnewlist = []
# for n in range(nbSamples):
#     newP = deepcopy(slist[n].p)
#     for m in range(0, nbFrames):
#         bTransform = np.random.rand(nbVar, 1) + 0.5
#         aTransform = np.random.rand(nbVar, nbVar) +0.5
#         for k in range(nbData):
#             newP[m, k].A = newP[m, k].A * aTransform
#             newP[m, k].b = newP[m, k].b #* bTransform
#     rnewlist.append(TPGMMGMR.reproduce(newP, slist[n].Data[1:nbVar, 0]))


ax =plt.subplot()
plt.xlim(-2,2)
plt.ylim(-2,2)
# plt.set_xlim(xlim)
# plt.set_ylim(ylim)
# plt.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
# plt.title('Reproductions with same task parameters')
for n in range(nbSamples):
    TPGMMGMR.plotReproduction(rnewlist[n], xaxis, yaxis, ax, showGaussians=True)   
plt.grid('on')
plt.show()