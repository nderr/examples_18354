import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

class InteractiveFigure:

    # spacing parameter
    N = 201
    nL = 41

    def __init__(self, L, c=1, zc=0):

        # length scale for plot
        self.L = L

        # radius of circle
        self.R  = np.abs(c - zc)
        self.zc = zc
        self.Gamma = 0
        self.alpha = 0
        self.c = c

        # Create a figure and axis
        # left axis is z-plane, right axis is w-plane
        self.fig, self.ax = plt.subplots(1,3,figsize=(12,5))

        # make room for several sliders on left side, for
        # Gamma, alpha, cx, cy
        self.fig.subplots_adjust(left=0.2, bottom=0.25)
        self.ax[0].set_title('z-plane')
        self.ax[1].set_title('Z-plane')
        self.ax[2].set_title('region mask')

        # remove all axis ticks and labels
        self.ax[1].set_xticks([])
        self.ax[1].set_yticks([])

        # add sliders, all horizontal, tucked on the left 
        # side of the figure
        self.ax_Gamma = self.fig.add_axes([0.05, 0.1, 0.1, 0.03])
        self.ax_alpha = self.fig.add_axes([0.05, 0.15, 0.1, 0.03])
        self.ax_cx = self.fig.add_axes([0.05, 0.2, 0.1, 0.03])
        self.ax_cy = self.fig.add_axes([0.05, 0.25, 0.1, 0.03])

        # set up sliders
        self.s_Gamma = plt.Slider(self.ax_Gamma, 'Gamma', -50, 0, valinit=0)
        self.s_alpha = plt.Slider(self.ax_alpha, 'alpha', -np.pi, np.pi, valinit=0)
        self.s_cx = plt.Slider(self.ax_cx, 'cx', -self.c, -1e-4, valinit=-0.2)
        self.s_cy = plt.Slider(self.ax_cy, 'cy', -self.c, self.c, valinit=0)

        # use plot_trafo as a handler for all sliders
        self.s_Gamma.on_changed(self.plot_trafo)
        self.s_alpha.on_changed(self.plot_trafo)
        self.s_cx.on_changed(self.plot_trafo)
        self.s_cy.on_changed(self.plot_trafo)

        # ranges
        xx = np.linspace(-self.L, self.L, InteractiveFigure.N)
        yy = np.linspace(-self.L, self.L, InteractiveFigure.N)
        XX,YY = np.meshgrid(xx,yy)

        # use complex variables
        self.ZZ = XX + 1j*YY

        self.plot_trafo()
        plt.show()


    def plot_trafo(self, event=None, **kwargs):

        # get the current values of the sliders
        self.Gamma = self.s_Gamma.val
        self.alpha = self.s_alpha.val
        self.zc = self.s_cx.val + 1j*self.s_cy.val
        self.R  = np.abs(self.c - self.zc)

        self.psi = Streamfunction(self.R, self.zc, self.Gamma, self.alpha, self.c)

        self.ax[0].set_facecolor('black')
        self.ax[1].set_facecolor('black')

        # contour levels
        levels = self.psi.psi_s + np.linspace(-5,5,InteractiveFigure.nL)

        # get streamline args with and w/o trafo
        x,y,psi_z = self.psi.streamplot_args(self.ZZ, axis='z')
        u,v,psi_w = self.psi.streamplot_args(self.ZZ, axis='w')

        # if contours already exist, remove them from the plot
        if hasattr(self, 'c1'):

            self.c1.remove()
            self.c2.remove()


        self.c1 = self.ax[0].contour(x,y,psi_z, levels=levels, **kwargs)
        mask,_,_ = self.psi.region_masks(self.ZZ)
        self.c2 = self.ax[1].contour(u,v,np.ma.array(psi_w,mask=mask) , levels=levels, **kwargs)

        if not hasattr(self,'im'):

            mask,below,above = self.psi.region_masks(self.ZZ)

            vv = np.zeros_like(self.ZZ,dtype=float)
            vv[mask] = 1
            vv[below] = 2
            vv[above] = 3

            self.im = self.ax[2].imshow(vv,origin='lower')

        else:

            mask,below,above = self.psi.region_masks(self.ZZ)

            vv = np.zeros_like(self.ZZ,dtype=float)
            vv[mask] = 1
            vv[below] = 2
            vv[above] = 3

            self.im.set_array(vv)

        self.ax[0].axis('square')
        self.ax[1].axis('square')

        if hasattr(self, 'patch1'): 
            self.patch1.remove()
            self.patch0.remove()

        self.patch0 = self.ax[0].add_patch(self.psi.p0)
        self.patch1 = self.ax[1].add_patch(self.psi.p1)

        if hasattr(self, 'stags_z'):

            s_z = self.psi.stags_z
            s_Z = self.psi.stags_Z
            si_z = self.psi.sings_z
            si_Z = self.psi.sings_Z

            self.stags_z.set_data(s_z[:,0],s_z[:,1])
            self.stags_Z.set_data(s_Z[:,0],s_Z[:,1])
            self.sings_z.set_data(si_z[:,0],si_z[:,1])
            self.sings_Z.set_data(si_Z[:,0],si_Z[:,1])

            # bring to front
            self.stags_z.set_zorder(3)
            self.stags_Z.set_zorder(3)
            self.sings_z.set_zorder(3)
            self.sings_Z.set_zorder(3)

        else:
            s_z = self.psi.stags_z
            s_Z = self.psi.stags_Z
            si_z = self.psi.sings_z
            si_Z = self.psi.sings_Z
        
            self.stags_z, = self.ax[0].plot(s_z[:,0],s_z[:,1], 'ro')
            self.stags_Z, = self.ax[1].plot(s_Z[:,0],s_Z[:,1], 'ro')
            self.sings_z, = self.ax[0].plot(si_z[:,0],si_z[:,1], 'rx', markersize=10)
            self.sings_Z, = self.ax[1].plot(si_Z[:,0],si_Z[:,1], 'rx', markersize=10)
        





class Streamfunction:

    def __init__(self, R, zc, Gamma, alpha, c):
        self.R     = R
        self.zc    = zc
        self.Gamma = Gamma
        self.alpha = alpha
        self.c     = c

        self.p0,self.p1,self.p0_pts,self.p1_pts = self.patches()
        self.stags_z,self.stags_Z, self.sings_z, self.sings_Z = self.pts()

    def pts(self):
    
        # stagnations pts in z-space
        asin = np.arcsin(self.Gamma/(4*np.pi*self.R))

        p1 = self.zc + self.R * np.exp(1j*(asin + self.alpha))
        p2 = self.zc + self.R * np.exp(1j*(np.pi-asin + self.alpha))

        # stagnation pts in w-space
        p3 = self.F(p1)
        p4 = self.F(p2)

        # make sure we get a streamline at trailing stag point
        self.psi_s = -self.Gamma*np.log(self.R) / (2*np.pi)

        # singular pts in z-space and w-space
        p5 = self.c
        p6 = -self.c
        p7 = self.F(p5)
        p8 = self.F(p6)


        # convert x + iy represenation to [x,y]
        p1 = np.array([p1.real, p1.imag])
        p2 = np.array([p2.real, p2.imag])
        p3 = np.array([p3.real, p3.imag])
        p4 = np.array([p4.real, p4.imag])

        p5 = np.array([p5.real, p5.imag])
        p6 = np.array([p6.real, p6.imag])
        p7 = np.array([p7.real, p7.imag])
        p8 = np.array([p8.real, p8.imag])


        return np.vstack((p1,p2)), np.vstack((p3,p4)), np.vstack((p5,p6)), np.vstack((p7,p8))


    def psi(self, z):

        zz = (z-self.zc) * np.exp(-1j*self.alpha)

        bad = np.logical_or(np.logical_or(np.isnan(zz), np.isinf(zz)), zz==0)
        out =  np.zeros_like(zz, dtype=complex)
        G_lz = np.zeros_like(zz, dtype=complex)

        mask, below, above = self.region_masks(z)

        G_lz[~bad] = self.Gamma * np.log(zz[~bad]) / (2*np.pi)
        out[~bad] = zz[~bad] + self.R**2/zz[~bad] - 1j*G_lz[~bad]
        out[bad] = np.nan
        return out

    def F(self, z):
        return z + self.c**2/z

    # detect whether we're directly above or under the airfoil
    def region_masks(self, z):
        xx = z.real
        yy = z.imag

        xvals = xx[0,:]
        yvals = yy[:,0]
        inds = np.arange(len(xvals))
        shell = np.zeros_like(z,dtype=bool)
        below = np.zeros_like(z,dtype=bool)
        above = np.zeros_like(z,dtype=bool)
        mask  = np.zeros_like(z,dtype=bool)

        # march through set of points defining the airfoil
        # if it crosses an x-coordinate, look for the next
        # y-coordinate farther away, and mark everything
        # past that y in that column (at that x) as above
        # or below as appropriate
        for i in range(len(self.p1_pts)-1):

            x0 = self.p1_pts[i][0]
            x1 = self.p1_pts[i+1][0]
            y0 = self.p1_pts[i][1]
            y1 = self.p1_pts[i+1][1]


            if x0 == x1:
                continue

            yv = lambda t : y0 + (y1-y0)*t

            # did we cross an x-coord?
            if x1 > x0:
                mx = np.logical_and(xvals > x0, xvals < x1)
            elif x1 < x0:
                mx = np.logical_and(xvals < x0, xvals > x1)
            else:
                continue

            # indices of crossed x-coords
            for j,x in zip(inds[mx],xvals[mx]):

                # get parametric t of intersection
                t = (x-x0)/(x1-x0)

                # get yvalue
                y = yv(t)

                if x1 > x0:
                    # CCW + incr. x => mark nearest yval below
                    dy = yvals - y

                    # want to take the last index where dy is negative
                    my = dy < 0
                    j0 = np.where(my)[0][-1]
                    shell[j0,j] = True

                else:
                    # CW - incr. x => mark nearest yval above
                    dy = yvals - y

                    # want to take the first index where dy is positive
                    my = dy > 0
                    j0 = np.where(my)[0][0]
                    shell[j0,j] = True

        for j in range(shell.shape[1]):

            if np.any(shell[:,j]):

                # get first nonzero index in mask
                bot = np.where(shell[:,j])[0][0]
                top = np.where(shell[:,j])[0][-1]

                below[:(bot+1),j] = True
                above[top:,j] = True
                mask[(bot+1):top,j] = True

        return mask, below, above

    # sqrt_function, choosing from possible solutions
    # to give us a continuous function around the airfoil
    # (effectively branch cut is hidden inside)
    def sqrt_fun(self, z):

        xx = z.real
        yy = z.imag

        ff = z**2 - 4*self.c**2
        rr = np.abs(ff)
        th = np.angle(ff)

        # branch cut along the negative real axis
        ss_1 = np.sqrt(rr) * np.exp(1j*th/2)
        ss_2 = np.sqrt(rr) * np.exp(1j*(th/2 + np.pi))

        # branch cut along positive real axis
        th[th < 0] += 2*np.pi
        ss_3 = np.sqrt(rr) * np.exp(1j*th/2)
        ss_4 = np.sqrt(rr) * np.exp(1j*(th/2 + np.pi))




        mask, below, above = self.region_masks(z)

        # quadrants
        q1 = np.logical_and(xx >= 0, yy > 0)
        q2 = np.logical_and(xx < 0, yy > 0)
        q3 = np.logical_and(xx < 0, yy <= 0)
        q4 = np.logical_and(xx >= 0, yy <= 0)

        # start with quadrants using the branch cut [-2,2]
        out = np.zeros_like(z, dtype=complex)
        out[q1] = ss_1[q1]
        out[q2] = ss_2[q2]
        out[q3] = ss_2[q3]
        out[q4] = ss_1[q4]

        # now paste over with any above/below
        below = np.logical_and(below, np.logical_and(yy >= 0, np.abs(xx) < 2))
        above = np.logical_and(above, np.logical_and(yy < 0, np.abs(xx) < 2))

        # plot to check and see how we're doing
        plot = False
        if plot:
            fig,ax = plt.subplots(1,4,figsize=(10,5))
            ax[0].imshow(below)
            ax[1].imshow(below)
            ax[2].imshow(above)
            ax[3].imshow(above)
            plt.show()

        out[below] = ss_4[below]
        out[above] = ss_3[above]

        out[mask] = np.nan + 1j*np.nan

        plot = False
        if plot:
            fig,ax = plt.subplots(2,5,figsize=(15,6))

            ax[0,0].set_title('[-2,2], +')
            ax[0,1].set_title('[-2,2], -')
            ax[0,2].set_title('[-inf,-2] U [2,inf], +')
            ax[0,3].set_title('[-inf,-2] U [2,inf], -')
            ax[0,4].set_title('inside body')

            plt.colorbar(ax[0,0].imshow(ss_1.real))
            plt.colorbar(ax[1,0].imshow(ss_1.imag))
            plt.colorbar(ax[0,1].imshow(ss_2.real))
            plt.colorbar(ax[1,1].imshow(ss_2.imag))
            plt.colorbar(ax[0,2].imshow(ss_3.real))
            plt.colorbar(ax[1,2].imshow(ss_3.imag))
            plt.colorbar(ax[0,3].imshow(ss_4.real))
            plt.colorbar(ax[1,3].imshow(ss_4.imag))
            plt.colorbar(ax[0,4].imshow(out.real))
            plt.colorbar(ax[1,4].imshow(out.imag))
            plt.show()


        # look for points above or below that need extension past the branch cut
        return out

    # inverse transform
    def Finv(self, Z):

        return 0.5*(Z + self.sqrt_fun(Z))

    def patches(self):

        # a white circle patch
        p0 = plt.Circle((self.zc.real, self.zc.imag), self.R, color='white', alpha=1, zorder=2)

        # a white patch corresponding to the circle under the mapping
        tt = np.linspace(0, 2*np.pi, InteractiveFigure.N)
        xx = self.zc.real + self.R * np.cos(tt)
        yy = self.zc.imag + self.R * np.sin(tt)
        ww = self.F(xx + 1j*yy)

        p0_pts = np.array([xx,yy]).T.tolist()
        p1_pts = np.array([ww.real,ww.imag]).T.tolist()

        

        wx = ww.real
        wy = ww.imag
        verts = np.array([wx,wy]).T.tolist()
        p1 = plt.Polygon(verts, color='white', alpha=1, zorder=2, linewidth=1.5)

        return p0,p1,p0_pts,p1_pts


    def streamplot_args(self, zz, axis):

        # make sure axis is either z or w
        if not (axis == 'z' or axis == 'w'):
            raise ValueError("axis must be either 'z' or 'w'")

        xx = zz.real
        yy = zz.imag
        pp = self.psi(zz).imag if axis == 'z' else self.psi(self.Finv(zz)).imag

        return xx,yy,pp

if __name__ == '__main__':

    L = 4

    InteractiveFigure(L)