import sys
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# functions for differentiating and integrating
from scipy.linalg import lu_factor, lu_solve
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron, eye, bmat
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter

def symbolic_solve():

    # Define the symbols
    u,v,s,d,k = sym.symbols('u v s d k', real=True)

    # Define the reaction vector
    R = sym.Array([
        u*(u*v - 1),
        s*(1 - v*u**2)
    ])

    # find the steady state(s)
    subs = sym.solve(R, (u,v), dict=True)
    sub = subs[0]

    # calculate jacobian and print at fixed point
    J = sym.Matrix([ R.diff(u), R.diff(v) ]).T.subs(sub)
    print()
    print('Jacobian:')
    print(f'    {J}')
    print()

    # compute the trace and determinant and print
    # at fixed point
    print('Trace:')
    print(f'    {J.trace()}')
    print()
    print('Determinant:')
    print(f'    {J.det()}')
    print()

    # compute the eigenvalues
    D_mat = -sym.diag(d,1) * k**2
    A = J + D_mat

    lam = []
    for eigenval,mult in A.eigenvals().items():
        lam.append(eigenval)

    print('Eigenvalues:')
    for l in lam:
        print(f'    => {l}')
    print()

    lam_funs = [sym.lambdify([d,s,k], sym.re(l), modules=['numpy']) for l in lam]
    R_fun = sym.lambdify([u,v,s], R, modules=['numpy'])

    return lam_funs, R_fun

class iwin:

    d0 = 1
    s0 = 1

    def __init__(self,k_max,ylims,l_funs,r_fun):

        self.kk = np.linspace(0,K,1000)

        self.l_funs = l_funs
        self.r_fun = r_fun

        self.init_plots()
        self.mk_sliders()
        self.mk_buttons()

    def init_plots(self):

        self.fig, ax = plt.subplots()
        self.line_z, = ax.plot(self.kk, np.zeros_like(self.kk), 'k--')
        self.line_ls = ax.plot(self.kk, np.transpose(
            [f(iwin.d0,iwin.s0,self.kk) for f in self.l_funs]
        ))

        ax.set_ylim(ylims)
        ax.set_xlim((0,K))

        ax.set_xlabel('k')
        ax.set_ylabel('Re($\lambda$)')

        self.fig.subplots_adjust(left=0.3, bottom=0.25)

    def mk_sliders(self):

        ax_d = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])
        ax_s = self.fig.add_axes([0.1, 0.25, 0.03, 0.65])

        self.d_slider = Slider(
            ax_d,
            'log d',
            -2.0,
            2.0,
            valinit=0.
        )

        self.s_slider = Slider(
            ax_s,
            's',
            0.,
            2.0,
            valinit=1.,
            orientation='vertical'
        )

        self.d_slider.on_changed(self.update)
        self.s_slider.on_changed(self.update)

    def mk_buttons(self):

        ax_rs = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        ax_sl = self.fig.add_axes([0.025, 0.025, 0.1, 0.04])

        self.rs_button = Button(
            ax_rs,
            'Reset',
            hovercolor='0.975'
        )

        self.sl_button = Button(
            ax_sl,
            'Solve',
            hovercolor='0.975'
        )

        self.rs_button.on_clicked(self.reset)
        self.sl_button.on_clicked(self.solve)


    def update(self,val):

        d = 10**self.d_slider.val
        s = self.s_slider.val

        self.line_z.set_ydata(np.zeros_like(self.kk))
        for line_l,f in zip(self.line_ls,self.l_funs):
            line_l.set_ydata(f(d,s,self.kk))

        self.fig.canvas.draw_idle()

    def reset(self,event):
        self.d_slider.reset()
        self.s_slider.reset()

    def solve(self,event):
        sol = solver(
                self.r_fun,
                10**self.d_slider.val,
                self.s_slider.val
        )

        sol.init_y()
        sol.init_fig()
        sol.solve(20,1e-2)

    def start(self):
        plt.show()

class solver:


    def __init__(self, R, d, s, N=96, L=50):

        self.N = N
        self.L = L
        self.d = d
        self.s = s

        self.R = lambda t,y: R(
                y[:N**2].reshape(N,N), # unpack u
                y[N**2:].reshape(N,N), # unpack v
                self.s
        )

        # spatial coordinates
        x = np.linspace(0, L, N, endpoint=False)  # Grid points
        y = np.linspace(0, L, N, endpoint=False)  # Grid points
        self.xx,self.yy = np.meshgrid(x,y)

        self.dx = L/N  # Grid spacing
        e = np.ones(N)

        # Construct a 1D Laplacian with periodic boundaries
        D1  = diags([e, -2*e, e], [-1, 0, 1], shape=(N, N))
        D1 += diags([e, e], [-N+1, N-1], shape=(N, N))  # Periodic boundary terms

        # 2D Laplacian with periodic BCs
        I = eye(N)
        self.M_ = (kron(I, D1) + kron(D1, I)) / (self.dx**2)
        self.I_ = eye(N**2)

        print(self.M_.shape)
        print(self.I_.shape)

        self.M = bmat([[self.d*self.M_, None], [None, self.M_]])

    def init_y(self):

        nu = np.random.randn(self.N**2).reshape(self.N,self.N)
        nv = np.random.randn(self.N**2).reshape(self.N,self.N)

        sig = 0.5*((2*np.pi/2.5) / self.L * self.N)
        nu = gaussian_filter(nu, sig, mode='wrap')
        nv = gaussian_filter(nv, sig, mode='wrap')

        nu -= nu.mean()
        nv -= nv.mean()

        u0 = (np.ones_like(nu) + nu).flatten()
        v0 = (np.ones_like(nv) + nv).flatten()

        self.y0 = np.concatenate((u0,v0))

    def update(self, i, y, dt):

        u_mi = y[:self.N**2].min()
        v_mi = y[self.N**2:].min()
        u_ma = y[:self.N**2].max()
        v_ma = y[self.N**2:].max()

        self.im_u.set_clim(u_mi,u_ma)
        self.im_v.set_clim(v_mi,v_ma)
        self.im_u.set_data(
                y[:self.N**2].reshape(self.N,self.N))
        self.im_v.set_data(
                y[self.N**2:].reshape(self.N,self.N)) 

        self.tt_u.set_text(f'u, t = {i*dt:.2f}')
        self.tt_v.set_text(f'v, t = {i*dt:.2f}')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def init_fig(self):

        self.fig, self.axs = plt.subplots(1,2,figsize=(10,5))

        # plot u
        self.im_u = self.axs[0].imshow(
            self.y0[:self.N**2].reshape(self.N,self.N),
            extent=(0,self.L,0,self.L),
        )  
        plt.colorbar(self.im_u, ax=self.axs[0])
        self.im_v = self.axs[1].imshow(
            self.y0[self.N**2:].reshape(self.N,self.N),
            extent=(0,self.L,0,self.L),
        )
        plt.colorbar(self.im_v, ax=self.axs[1])

        self.tt_u = self.axs[0].set_title('u, t = 0')
        self.tt_v = self.axs[1].set_title('v, t = 0')

        plt.ion()
        plt.show(block=False)

    def solve(self,T,dt):

        print('')
        print('starting solve')
        print('factorizing matrix...',end='')
        sys.stdout.flush()
        self.lup_u = lu_factor((self.I_ - 0.5*dt*self.d*self.M_).toarray(),overwrite_a=False)
        self.lup_v = lu_factor((self.I_ - 0.5*dt*self.M_).toarray(),overwrite_a=False)
        print('done')

        y = self.y0
        ns = int(T/dt)
        rem = T - ns*dt
        dt += rem/ns

        for i in range(ns):
            print(i)

            y = self.step(y,dt)
            self.update(i,y,dt)


    def step(self,y,dt):

        b = solve_ivp(
            self.R,
            (0,dt),
            y,
            method='RK45',
            t_eval=[dt],
            vectorized=True
        ).y[:,-1] + 0.5 * dt * self.M @ y

        bu = b[:self.N**2]
        bv = b[self.N**2:]

        lu_solve(self.lup_u, bu, overwrite_b=True)
        lu_solve(self.lup_v, bv, overwrite_b=True)

        return np.concatenate((bu,bv))
        

if __name__ == '__main__':

    L,R = symbolic_solve()

    K = 5
    ylims = (-5,1)

    fig = iwin(K,ylims,L,R)

    input('\nPress enter to start')
    print()
    fig.start()

