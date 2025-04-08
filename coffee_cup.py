import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import jn_zeros, j0, j1

T = 1
R = 1
Om = 1
nu = 1

Nt = 1000
dt = 1/Nt
Nr = 100
Nb_max = 100

rr = np.linspace(0, R, Nr)

# precompute Bessel function shapes
lam = jn_zeros(1, Nb_max)[np.newaxis, :] # (1, Nb_max)
B = j1(lam * rr[:,np.newaxis] / R)       # (Nr, Nb_max)
AB = ((-2*Om*R / (lam * j0(lam))) * B)   # (Nr, Nb_max)

# lambda for time dependence
exp_t = lambda nu,t : np.exp(-nu * t * lam**2 / R**2) # (1, Nb_max)

def u(nu,t,n):

    return np.sum(
        AB[:, :n] * exp_t(nu,t)[:, :n],
        axis=1
    )

fig,ax = plt.subplots()

# adjust plot to make room for slider (1 on bottom, 2 on left)
fig.subplots_adjust(left=0.25, bottom=0.25)

# add in axis for slider on bottom (for t) and two on the left (for nu and N)
ax_t = fig.add_axes([0.25, 0.1, 0.65, 0.025])
ax_nu = fig.add_axes([0.05, 0.1, 0.025, 0.8])
ax_N = fig.add_axes([0.1, 0.1, 0.025, 0.8])

# make the sliders
nu_slider = Slider(
    ax_nu,
    'log(nu)',
    -2,
    2,
    valinit=0.,
    valstep=0.1,
    orientation='vertical',
)

t_slider = Slider(
    ax_t,
    't',
    0,
    T,
    valinit=dt,
    valstep=dt
)

N_slider = Slider(
    ax_N,
    'N',
    1,
    Nb_max,
    valinit=10,
    valstep=1,
    orientation='vertical'
)

# make the initial plot
line, = ax.plot(rr, u(nu_slider.val, t_slider.val, N_slider.val), lw=2)
ax.set_xlabel('radial distance r')
ax.set_ylabel('u(r,t)')
t_str = lambda t: f'time t = {t:.2f}'
ax.set_title(t_str(0))

def update(nu, t, N):

    # update the line data 
    line.set_ydata(u(nu, t, N))
    ax.set_title(t_str(t))
    fig.canvas.draw_idle()

# link to handlers
nu_slider.on_changed(lambda log_nu: update(10**log_nu, t_slider.val, N_slider.val))
N_slider.on_changed( lambda N : update(10**nu_slider.val, t_slider.val, N))
t_slider.on_changed( lambda t : update(10**nu_slider.val, t, N_slider.val))

plt.show()