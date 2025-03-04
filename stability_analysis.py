#%%

####!%matplotlib ipympl

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

#plt.rcParams['text.usetex'] = True

# allow for animation plots using 

# %% [markdown]
#
# # Stability Analysis
#
# This is an example of an equation
# $$y = x^2$$
# 
#
#
# %%

xx = np.linspace(-5,5,100)
yy = lambda om,phi: np.sin(om*(xx + phi))

fig, ax = plt.subplots()

om_0 = 1
phi_0 = 0
line, = ax.plot(xx,yy(om_0,phi_0))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')

#fig.subplots_adjust(left=0.25, bottom=0.25)

#ax_om = fig.add_axes([0.25, 0.1, 0.65, 0.03])

plt.show()
# %%
