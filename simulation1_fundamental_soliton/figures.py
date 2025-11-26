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

# Set global font sizes - SAME AS PREVIOUS VERSION
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    'figure.titlesize': 24
})

# -- CONVENIENT ABBREVIATIONS
FT = nfft.ifft
IFT = nfft.fft


def figure_1a(z,t, u, tLim=None ,wLim=None, oName=None):
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
        # Increased vertical spacing to prevent label overlap
        cax = f.add_axes([x0, y0+1.05*h, w, 0.03*h])
        cbar = f.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(color='k',
                            labelcolor='k',
                            bottom=False,
                            direction='out',
                            labelbottom=False,
                            labeltop=True,
                            top=True,
                            size=6,
                            pad=2,
                            labelsize=16  # Larger tick labels
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

    # LARGER FIGURE SIZE - SAME AS PREVIOUS
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(18, 10))
    
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
    # Larger font size for colorbar title - SAME AS PREVIOUS
    cbar1.ax.set_title(r"$|A|^2$ (normalized)", color='k', y=2.5, pad=10, fontsize=22)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xlim(tLim)
    ax1.set_ylim([0.,z.max()])
    ax1.set_xlabel(r"Time $t$", fontsize=24)
    ax1.set_ylabel(r"Propagation distance $z$", fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    # -- RIGHT SUB-FIGURE: ANGULAR FREQUENCY-DOMAIN PROPAGATION CHARACTERISTICS 
    Iw = np.abs(nfft.ifftshift(FT(u, axis=-1),axes=-1))**2
    Iw /= np.max(Iw[0])
    Iw = _truncate(Iw)
    im2 = ax2.pcolorfast(w,z,Iw[:-1,:-1],
                         norm=col.LogNorm(vmin=Iw.min(),vmax=Iw.max()),
                         cmap=cmap
                         )
    cbar2 =_setColorbar(im2,ax2.get_position())
    # Larger font size for colorbar title - SAME AS PREVIOUS
    cbar2.ax.set_title(r"$|A_\omega|^2$ (normalized)", color='k', y=2.5, pad=10, fontsize=22)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.set_xlim(wLim)
    ax2.set_ylim([0.,z.max()])
    ax2.set_xlabel(r"Angular frequency $\omega$", fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(labelleft=False)

    if oName:
        # HIGH RESOLUTION - increased to 2400 DPI
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

    f, ax = plt.subplots(figsize=(12, 10))
    ax.plot(dz, RMSError_1, r"o-", label=r"simple splitting", markersize=12, linewidth=3)
    ax.plot(dz, RMSError_2, r"^-", label=r"symmetric splitting", markersize=12, linewidth=3)
    ax.set(xlabel=r"stepsize $dz$",ylabel=r"RMS error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which='both', ls='-', color='0.65')
    ax.legend(fontsize=22)
    
    # Larger label and tick font sizes - SAME AS PREVIOUS
    ax.set_xlabel(r"stepsize $dz$", fontsize=26)
    ax.set_ylabel(r"RMS error", fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=22)

    if oName:
        # HIGH RESOLUTION - increased to 2400 DPI
        plt.savefig(oName,format='png',dpi=600, bbox_inches='tight')
    else:
        plt.show()


figure_1c = figure_1a
figure_2a = figure_1a

# EOF: figures.py