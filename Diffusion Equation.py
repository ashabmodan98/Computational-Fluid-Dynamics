import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D	

def Toldfn(i,j,T):
    if (i==-1) and (j!=-1) and (j!=Ny):
        return T[1,j]
    elif (j==-1) and (i!=-1) and (j!=Nx):
        return T[i,1]
    elif (i==Nx) and (j!=-1) and (j!=Ny):
        return T[Nx-2,j]
    elif (j==Ny) and (i!=-1) and (i!=Nx):
        return T[i,Ny-2]
    else:
        return T[i,j]


xf = 1
Nx = 60
dx = xf/(Nx)
yf = 1
Ny = 60
dy = yf/(Ny)
Nt = 80
dt = 0.001

x = np.arange(0,xf,dx)
y = np.arange(0,yf,dy)
X,Y = np.meshgrid(x,y)

extent=(0,60,0,60)
levels = np.arange(0,100,15)

T = np.zeros([Nx,Ny])
Tnew = np.array(T)
#Told = np.array(T)
Told1 = np.array(T)

a = 3*Nx/5

for i in range(0,Nx-1):  #initial condition
    for j in range(0,Ny-1):
        if j <= (a-i):
            Tnew[i,j] = 100
            
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X,Y,Tnew,cmap = 'plasma',edgecolor='none')
ax.set_title('Initial Conditions')
ax.set_zlabel('Temperature')
plt.show()

Told1 = Tnew

ux = math.cos(math.pi/4)
uy = math.cos(math.pi/4)

alpha = 0.01
rx = alpha*dt/dx**2
ry = alpha*dt/dy**2

for n in range (Nt):
    for i in range (Nx):
        for j in range (Ny):
            if i==0 or i==Nx-1:
                Tnew[i,j] = rx * (Toldfn(i+1,j,Told1) - 2*Toldfn(i,j,Told1) + Toldfn(i-1,j,Told1)) \
                          + ry * (Toldfn(i,j+1,Told1) - 2*Toldfn(i,j,Told1) + Toldfn(i,j-1,Told1))\
                          - uy * (Toldfn(i,j,Told1) - Toldfn(i,j-1,Told1))* dt/dy + Toldfn(i,j,Told1)
            elif j==0 or j==Ny-1:
                Tnew[i,j] = rx * (Toldfn(i+1,j,Told1) - 2*Toldfn(i,j,Told1) + Toldfn(i-1,j,Told1)) \
                          + ry * (Toldfn(i,j+1,Told1) - 2*Toldfn(i,j,Told1) + Toldfn(i,j-1,Told1))\
                          - ux * (Toldfn(i,j,Told1) - Toldfn(i-1,j,Told1))* dt/dx + Toldfn(i,j,Told1)
            else:
                Tnew[i,j] = rx * (Toldfn(i+1,j,Told1) - 2*Toldfn(i,j,Told1) + Toldfn(i-1,j,Told1))\
                          + ry * (Toldfn(i,j+1,Told1) - 2*Toldfn(i,j,Told1) + Toldfn(i,j-1,Told1))\
                          - uy * (Toldfn(i,j,Told1) - Toldfn(i,j-1,Told1))* dt/dy\
                          - ux * (Toldfn(i,j,Told1) - Toldfn(i-1,j,Told1))* dt/dx + Toldfn(i,j,Told1)
    Told1=Tnew

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X,Y,Told1,cmap = 'plasma',edgecolor='none')
ax.set_zlabel('Temperature')
ax.set_title('Temp. diffusion')
plt.show()

c = plt.contourf(Told1, levels)
plt.imshow(Told1, origin='lower', cmap='plasma')
CS = plt.contour(Told1, levels, colors='k', origin='lower', extent=extent)
plt.clabel(CS, inline=1, fontsize=10)
plt.colorbar(c)
plt.show()

