import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

# Lotka-Volterra model
#
# du/dt = a*u - b*u*v
# dv/dt = -c*v + e*u*v
#
# y = [u,v]
def f(a,b,c,e,t,y):
    # unused t argument is required by solve_ivp

    return [
        a*y[0] - b*y[0]*y[1],
        -c*y[1] + e*y[0]*y[1]
    ]

# pick some parameters
a = 1.
b = 1.
c = 1.
e = 1.
fun = partial(f, a, b, c, e) # now y_dot = fun(t, y)

# simulate from t = 0 to 50, start at u = 2, v = 0.1
t_span = (0, 50)
y0 = [2, 0.1]

# solve ODE
sol = solve_ivp(
        fun,
        t_span,
        y0,
        t_eval=np.linspace(*t_span, 1000),
        method='DOP853'
)

# plot results
plt.plot(sol.t, sol.y[0], label='prey')
plt.plot(sol.t, sol.y[1], label='predator')
plt.legend()
plt.xlabel('time')
plt.ylabel('population')

plt.show()

# plot trajectory through phase space
plt.plot(sol.y[0], sol.y[1])
plt.xlabel('prey (u)')
plt.ylabel('predator (v)')
plt.show()

