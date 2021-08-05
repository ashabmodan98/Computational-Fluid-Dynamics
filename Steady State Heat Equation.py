import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D	

Nx = 30
Ny = 30
dx = 1/Nx
dy = 1/Ny
x = np.arange(0,1,dx)
y = np.arange(0,1,dy)

T = np.ones([Nx,Ny])

#boundary conditions
T[:,0] = 0
T[:,-1] = 100
T[0,:Nx//2] = 100
T[0,Nx//2:0] = 0
T[-1,:] = 0

Told = np.array(T)
T_ini = T

denom = 2*((dx**2 + dy**2)/(dx*dy)**2)
error = 1e-4
tolerance = 1e-5
iterr = 0
w = 1.2
#w_list=[1+0.1*i for i in range(10)]
iterr_list = []

extent=(0,30,0,30)
levels = np.arange(0,90,7)

while(error > tolerance):
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            
            T[i,j] = T[i,j]*(1-w) + w*((T[i-1,j] + Told[i+1,j])/dx**2 + (T[i,j-1] + Told[i,j+1])/dy**2)/denom
    error = np.max(abs(T-Told))

    Told = np.array(T)
    iterr = iterr + 1
iterr_list.append(iterr)
print('The convergence criterion was fulfilled at the ',iterr+1,'th iteration for the values, w = ',w, 'and tolerance = ',tolerance)
c = plt.contourf(T, levels)
plt.imshow(T, origin='lower', cmap='plasma')
CS = plt.contour(T, levels, colors='k', origin='lower', extent=extent)
plt.clabel(CS, inline=1, fontsize=10)
plt.colorbar(c)

X,Y = np.meshgrid(x,y,sparse=True)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X,Y,T,cmap = 'plasma',edgecolor='none')
ax.set_title('Temperature')
#plt.plot(iterr_list,w_list)
plt.show()
