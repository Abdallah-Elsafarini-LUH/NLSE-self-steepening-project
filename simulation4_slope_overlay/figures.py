"""figures.py

module implementing the figures used during exercise session 04

Supplementary material for the lecture "Computational Photonics" held at
Leibniz University Hannover in summer term 2017

AUTHOR: OM
DATE: 2020-06-01
"""
import numpy as np
import numpy.fft as nfft
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col

# Set global font sizes - EXTRA LARGE
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 28,
    'axes.labelsize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 24,
    'figure.titlesize': 28
})

# -- CONVENIENT ABBREVIATIONS
FT = nfft.ifft
IFT = nfft.fft


def figure_1a(z,t, u, s, tLim=None ,wLim=None, oName=None):
    """Plot pulse propagation scene

    Generates a plot showing the z-propagation characteristics of
    the squared magnitude field envelope (left subfigure) and
    the spectral intensity (right subfigure).

    Args:
        z (array): samples along propagation distance
        t (array): time samples
        u (array): time domain field envelope
        tLim (2-tuple): time range in the form (tMin,tMax)
                        (optional, default=None)
        wLim (2-tuple): angular frequency range in the form (wMin,wMax)
                        (optional, default=None)
        oName (str): name of output figure
                        (optional, default: None)
    """

    def _setColorbar(im, refPos):
        """colorbar helper"""
        x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
        cax = f.add_axes([x0, y0+1.02*h, w, 0.02*h])
        cbar = f.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(color='k',
                            labelcolor='k',
                            bottom=False,
                            direction='out',
                            labelbottom=False,
                            labeltop=True,
                            top=True,
                            size=8,
                            pad=2,
                            labelsize=20  # Larger tick labels
                            )

        cbar.ax.tick_params(which="minor", bottom=False, top=False )
        return cbar

    def _truncate(I):
        """truncate intensity

        fixes python3 matplotlib issue with representing small
        intensities on plots with log-colorscale
        """
        I[I<1e-6]=1e-6
        return I

    w = nfft.ifftshift(nfft.fftfreq(t.size,d=t[1]-t[0])*2*np.pi)

    if tLim==None:
       tLim = (np.min(t),np.max(t))
    if wLim==None:
       wLim = (np.min(w),np.max(w))

    # LARGER FIGURE SIZE
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 12))
    
    cmap=mpl.cm.get_cmap('jet')

    # -- LEFT SUB-FIGURE: TIME-DOMAIN PROPAGATION CHARACTERISTICS
    It = np.abs(u)**2
    It /= np.max(It[0])
    It = _truncate(It)
    im1 = ax1.pcolorfast(t, z, It[:-1,:-1],
                         norm=col.LogNorm(vmin=It.min(),vmax=It.max()),
                         cmap=cmap
                         )
    cbar1 = _setColorbar(im1,ax1.get_position())
    # Larger font size for colorbar title
    cbar1.ax.set_title(r"$|A|^2$ (normalized)", color='k', y=3.5, pad=10, fontsize=26)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xlim(tLim)
    ax1.set_ylim([0.,z.max()])
    ax1.set_xlabel(r"Time $t$", fontsize=28)
    ax1.set_ylabel(r"Propagation distance $z$", fontsize=28)
    ax1.tick_params(axis='both', which='major', labelsize=24)
    
    I0 = np.max(np.abs(u[0])**2)
    tc = s * I0 * z
    # MUCH THICKER DOTTED LINE
    ax1.plot(tc, z, color='magenta', dashes=[2,1], linewidth=5, label="$t_c(z)$")

    # -- RIGHT SUB-FIGURE: ANGULAR FREQUENCY-DOMAIN PROPAGATION CHARACTERISTICS 
    Iw = np.abs(nfft.ifftshift(FT(u, axis=-1),axes=-1))**2
    Iw /= np.max(Iw[0])
    Iw = _truncate(Iw)
    im2 = ax2.pcolorfast(w,z,Iw[:-1,:-1],
                         norm=col.LogNorm(vmin=Iw.min(),vmax=Iw.max()),
                         cmap=cmap
                         )
    cbar2 =_setColorbar(im2,ax2.get_position())
    # Larger font size for colorbar title
    cbar2.ax.set_title(r"$|A_\omega|^2$ (normalized)", color='k', y=3.5, pad=10, fontsize=26)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.set_xlim(wLim)
    ax2.set_ylim([0.,z.max()])
    ax2.set_xlabel(r"Angular frequency $\omega$", fontsize=28)
    ax2.tick_params(axis='both', which='major', labelsize=24)
    ax2.tick_params(labelleft=False)

    if oName:
        # HIGH RESOLUTION
        plt.savefig(oName, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()


def figure_1b(res,oName=None):
    """Plot RMS error of splitting schemes

    Generates a loglog-plot showing the scaling behavior of the
    root-mean-square error at given z-stepsize for the
    simple and symmetric operator splitting schemes.

    Args:
        res (array): results of the simulation run in main_b of ex04 part 1
        oName (str): name of output figure (optional, default: None)
    """

    dz, RMSError_1, RMSError_2 = zip(*res)

    f, ax = plt.subplots(figsize=(14, 12))
    ax.plot(dz, RMSError_1, r"o-", label=r"simple splitting", markersize=14, linewidth=4)
    ax.plot(dz, RMSError_2, r"^-", label=r"symmetric splitting", markersize=14, linewidth=4)
    ax.set(xlabel=r"stepsize $dz$",ylabel=r"RMS error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which='both', ls='-', color='0.65')
    ax.legend(fontsize=26)
    
    # Very large label and tick font sizes
    ax.set_xlabel(r"stepsize $dz$", fontsize=30)
    ax.set_ylabel(r"RMS error", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=26)

    if oName:
        # HIGH RESOLUTION
        plt.savefig(oName,format='png',dpi=600, bbox_inches='tight')
    else:
        plt.show()


figure_1c = figure_1a
figure_2a = figure_1a

# EOF: figures.py