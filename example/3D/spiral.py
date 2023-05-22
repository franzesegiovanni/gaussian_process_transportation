#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import turtle as t
from matplotlib import cm

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib.ticker import LinearLocator
import pathlib
# %matplotlib qt5
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
		'''
		Adapted and modifed to get the unknowns for defining a parabola:
		http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
		'''

		denom = (x1-x2) * (x1-x3) * (x2-x3)
		A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
		B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
		C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

		return A,B,C


z_pos=[]
x_pos=[]
y_pos=[]

t.tracer(10,1)

for i in range(360):
    t.circle(i,3)
    # time.sleep(0.1)
    x_pos.append(0.01*t.pos()[0])
    y_pos.append(0.01*t.pos()[1])
    z_pos.append(0)
    # print(t.pos())
t.update()
t.bye()

#Define your three known points
s0,z0=[0,0]
s1,z1=[1,0]
sc,zc=[0.5,1]


x0,x1=[x_pos[-1],x_pos[0]]
y0,y1=[y_pos[-1],y_pos[0]]
#Calculate the unknowns of the equation y=ax^2+bx+c
a,b,c=calc_parabola_vertex(s0, z0, sc, zc, s1, z1)
#Define x range for which to calc parabola
s_pos=np.arange(0,1,0.01)
# print(s_pos)
#Calculate y values 
for s in range(len(s_pos)):
    s_val=s_pos[s]
    z=(a*(s_val**2))+(b*s_val)+c
    x= (1-s_val)*x0 + s_val*x1
    y= (1-s_val)*y0 + s_val*y1
    x_pos.append(x)
    y_pos.append(y)
    z_pos.append(z)

 
# Plot the parabola (+ the known points)
fig = plt.figure() 

# syntax for 3-D projection
ax = plt.axes(projection ='3d')
ax.set_xlim(min(x_pos), max(x_pos))
ax.set_ylim(min(y_pos), max(y_pos))
ax.set_zlim(-max(z_pos), max(z_pos))
# plt.plot(s_pos, z_pos, linestyle='-.', color='black') # parabola line
# ax.scatter(x_pos, y_pos, z_pos, color='red') # parabola points
# plt.scatter(x1,y1,color='r',marker="D",s=50) # 1st known xy
# plt.scatter(x2,y2,color='g',marker="D",s=50) # 2nd known xy
# plt.scatter(x3,y3,color='k',marker="D",s=50) # 3rd known xy
# plt.show()

print(min(z_pos))
print(max(z_pos))

x = np.arange(min(x_pos), max(x_pos), 0.125)
# x = np.arange(1, 10, 10)

y = np.arange(min(y_pos), max(y_pos), 0.125)
# print('shape of x',x.shape)
# print('shape of y',y.shape)
X, Y = np.meshgrid(x, y)
Z = 0*X
# print('shape of Z',Z.shape)



X_=X.reshape(-1,1)
Y_=Y.reshape(-1,1)
X_train=np.hstack([X_,Y_])

k=C(constant_value=0.01)*RBF(1*np.random.rand(2,1))

K=k(X_train,X_train)+0.0001*np.eye(X_train.shape[0])
L = np.linalg.cholesky(K)

u = np.random.normal(loc=0, scale=1, size=len(X_train))

Znew = np.dot(L, u)


Znew=Znew.reshape(X.shape)
# Plot the surface.
# print('shape of X',X.shape)
print(X_.shape)
print(Znew.shape)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

newsurf = ax.plot_surface(X, Y, Znew, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                    #    linewidth=0, antialiased=False)


ax.scatter(x_pos, y_pos, z_pos, color='red') # parabola points



plt.show()

# Save

# demo = np.array([x_pos,y_pos,z_pos]).T
# old_surface = np.array([X,Y,Z]).T
# new_surface = np.array([X,Y,Znew]).T

# np.savez(str(pathlib.Path().resolve())+'/data/'+str('last_spiral')+'.npz', 
#         demo=demo, old_surface=old_surface,new_surface=new_surface) 


# print("demo, surface, new surface saved")





## Reproduce/Load
# fig = plt.figure()

# # syntax for 3-D projection
# ax = plt.axes(projection ='3d')
# ax.set_xlim(min(x_pos), max(x_pos))
# ax.set_ylim(min(y_pos), max(y_pos))
# ax.set_zlim(min(z_pos), max(z_pos))
# newsurf = ax.plot_surface(new_surface[:,:,0], new_surface[:,:,1], new_surface[:,:,2], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# # surf = ax.plot_surface(old_surface[:,:,0], old_surface[:,:,1], old_surface[:,:,2], cmap=cm.coolwarm,
#                     #    linewidth=0, antialiased=False)



# ax.scatter(x_pos, y_pos, z_pos, color='red') # parabola points

# %%
