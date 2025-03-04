#%%

#!%matplotlib ipympl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.widgets import Slider, Button
import shutil

use_tex = False
if use_tex:
    plt.rcParams['text.usetex'] = True if shutil.which('latex') else False
else:
    plt.rcParams['text.usetex'] = False

#%%

# define domain and plotted function
xx = np.linspace(-5,5,1000)
yy = lambda om,ph: np.sin(om*(xx + ph))

#%%

# create figure and axes
fig, ax = plt.subplots()

# plot initial values
om_0 = 1
ph_0 = 0
line, = ax.plot(xx,yy(om_0,ph_0))

# set labels
if plt.rcParams['text.usetex']:
    x_lab = r'$x$'
    y_lab = r'$\sin\left[\omega\left(x-\phi\right)\right]$'
    om_lab = r'$\omega$'
    ph_lab = r'$\phi$'
else:  
    x_lab = 'x'
    y_lab = 'sin[ω(x-φ)]'
    om_lab = 'ω'
    ph_lab = 'φ'

# create labels
plt.xlabel(x_lab)
plt.ylabel(y_lab)

# add slider and button axes
fig.subplots_adjust(left=0.3, bottom=0.25)
ax_om = fig.add_axes([0.25, 0.1, 0.65, 0.03])
ax_ph = fig.add_axes([0.1, 0.25, 0.03, 0.65])
ax_bt = fig.add_axes([0.8, 0.025, 0.1, 0.04])

# create sliders and button
om_slider = Slider(
    ax_om,
    om_lab,
    0.1,
    10.0,
    valinit=om_0
)

ph_slider = Slider(
    ax_ph,
    ph_lab,
    -np.pi,
    np.pi,
    valinit=ph_0,
    orientation='vertical'
)

rs_button = Button(
    ax_bt,
    'Reset',
    hovercolor='0.975'
)

# update and reset function
def update(val):
    om = om_slider.val
    ph = ph_slider.val
    line.set_ydata(yy(om,ph))
    fig.canvas.draw_idle()

def reset(event):
    om_slider.reset()
    ph_slider.reset()

# connect sliders and button to update and reset functions
om_slider.on_changed(update)
ph_slider.on_changed(update)
rs_button.on_clicked(reset)

plt.show()