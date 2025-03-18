# big imports
import sys
import tqdm
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt

# functions for differentiating and integrating
from scipy.linalg import lu_factor, lu_solve
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve

# grab the arguments from the command line
parser = ap.ArgumentParser(description='Swift-Hohenberg simulation')
parser.add_argument('-N', type=int,   default=96,  help='Grid size NxN')
parser.add_argument('-r', type=float, default=0.1, help='r parameter')
parser.add_argument('-g', type=float, default=0.0, help='g parameter')
parser.add_argument('-t', type=float, default=0.5, help='Time step')
parser.add_argument('-T', type=float, default=100, help='Duration')
parser.add_argument('-L', type=float, default=50,  help='Domain size')

args = parser.parse_args()

# %%
N = args.N
r = args.r
g = args.g
dt = args.t
T = args.T
L = args.L

# print what we're doing
print('---')
print(f'solving Swift-Hohenberg equation')
print(f'  parameters:')
print(f'    grid size :  {N}x{N}')
print(f'    domain    :  [0,{L}]x[0,{L}]')
print(f'    r         :  {r}')
print(f'    g         :  {g}')
print(f'    dt        :  {dt}')
print(f'    T         :  {T}') 
print('---')

# we'll use a direct solve if N < 196
direct = N < 196

# spatial coordinates
x = np.linspace(0, L, N, endpoint=False)  # Grid points
y = np.linspace(0, L, N, endpoint=False)  # Grid points
xx,yy = np.meshgrid(x,y)
dx = L/N  # Grid spacing
e = np.ones(N)

# Construct a 1D Laplacian with periodic boundaries
D1  = diags([e, -2*e, e], [-1, 0, 1], shape=(N, N))
D1 += diags([e, e], [-N+1, N-1], shape=(N, N))  # Periodic boundary terms

# 2D Laplacian with periodic BCs
I = eye(N)
M = kron(I, D1) + kron(D1, I)
I = eye(N**2)

# construct the laplacian + bilaplacian sum
A = -2*M/(dx**2) - (M @ M) / (dx**4)



if direct:

    print('---')
    print('factorizing matrix for direct solve...',end='')
    sys.stdout.flush()

    lu,piv = lu_factor((I-0.5*dt*A).toarray(),overwrite_a=False)

    print('done')
    print('---')

def step(y):

    # u_n+1 = u_n + dt* ((r-1)u_n + gu_n^2 - u_n^3)
    #          + 0.5*dt*A*(u_n + u_n+1)

    # solve the nonspatially dependent parts with canned ODE solver
    # instead of forward Euler above
    b = solve_ivp(
            lambda t,u : (r-1)*u + g*u**2 - u**3,
            [0,dt],
            y,
            t_eval=[dt],
            vectorized=True
        ).y[:,-1] + 0.5*dt*A@y

    # now solve the linear part with Crank-Nicolson
    if direct:
        lu_solve((lu,piv),b,overwrite_b=True)
        return b
    else:
        return spsolve(I-0.5*dt*A,b)

#%%

# helper function to create gaussian humps
def gauss(x0,y0,sig):

    nx = xx/L
    ny = yy/L
    zz_sq = ((nx-x0)**2 + (ny-y0)**2) / sig**2
    return np.exp(-0.5*zz_sq)

# normalize spatial coordinates
px = 2*np.pi*xx/L
py = 2*np.pi*yy/L

# initial condition: some hills + background wave
wid = 0.02
p0 = 0.05*(np.sin(px) + np.cos(py) + \
    + np.sin(5*px) + np.cos(5*py)) + \
    + gauss(.20,.20,wid) + gauss(.80,.70,wid) + \
    + gauss(.40,.50,wid) + gauss(.30,.70,wid) + \
    + gauss(.70,.25,wid) + gauss(.30,.30,wid)
p0 = p0.flatten()

# set up plot
fig,ax = plt.subplots()
im = ax.imshow(p0.reshape((N,N)))
plt.colorbar(im)
tt = ax.set_title('t = 0')

plt.ion()
plt.show()

print('')
input('Press enter to start time-stepping')
print('')

# get ready for time-stepping
p = p0
t = 0
n_steps = int(T/dt)
rem = T - n_steps*dt
dt += rem/n_steps

# jump in
for k in tqdm.tqdm(range(1,n_steps+1)):

    # take our step
    p = step(p)

    # plot the result
    ma = np.max(p)
    mi = np.min(p)
    im.set_clim(mi,ma)
    im.set_data(p.reshape((N,N)))
    tt.set_text(f't = {k*dt}')

    fig.canvas.draw()
    fig.canvas.flush_events()

print('')
input('Press enter to close')