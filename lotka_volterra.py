import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
from functools import partial


def f(a,b,c,e,t,y):

    return [
        a*y[0] - b*y[0]*y[1],
        -c*y[1] + e*y[0]*y[1]
    ]


a = 1.
b = 1.
c = 1.
e = 1.
fun = partial(f, a, b, c, e)

t_span = (0, 50)
y0 = [2, 0.1]

sol = solve_ivp(
        fun,
        t_span,
        y0,
        t_eval=np.linspace(*t_span, 1000),
        method='DOP853'
)

plt.plot(sol.t, sol.y[0], label='prey')
plt.plot(sol.t, sol.y[1], label='predator')
plt.legend()
plt.xlabel('time')
plt.ylabel('population')

plt.show()

plt.plot(sol.y[0], sol.y[1])
plt.xlabel('prey (u)')
plt.ylabel('predator (v)')
plt.show()

