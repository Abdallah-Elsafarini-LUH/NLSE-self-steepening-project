"""
Computer Exercises for Lecture "Computational Photonics"
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
from figures import figure_2a
from split_step_solver import SSFM_HONSE_symmetric

def figure_pulse_shape(z, t, Azt, tLim = (-5.5,5.5)): 
    f, ax = plt.subplots() 
    
    I = np.abs(Azt)**2
    
    _z2id = lambda z0: np.argmin(np.abs(z-z0))
    
    ax.plot(t, I[_z2id(10.0), :], color = 'k', dashes = [], label = "10")
    ax.plot(t, I[_z2id(5.0), :], color = 'k', dashes = [3,1], label = "5")
    ax.plot(t, I[_z2id(0.0), :], color = 'k', dashes = [1,1], label = "0")
    
    ax.set_xlim(tLim)
    ax.set_xlabel("Time $t$")
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Intensity $I$")

    ax.legend(
        title="$z/L_{D}$",    
        loc = "upper left"
    )
    
    plt.show()

def main_a():

    # -- SET FIBER PARAMETERS
    beta2 = -1 #-0.01276                        # (ps^2/m)
    beta3 = 0 #8.119*1e-5                     # (ps^3/m)
    beta4 = 0 #-1.321*1e-7                     # (ps^4/m)
    gamma = 1 #0.045                           # (1/W/m) nonlinear coefficient
    s = 0.2                                     # self-steepening parameter
   
    # -- SET PULSE PARAMETERS
    t0 = 1 #0.0284                             # (ps) pulse duration
    P0 = np.abs(beta2)/t0/t0/gamma          # (W) pulse peak power
   
    # -- SET PARAMETERS FOR COMPUTATIONAL DOMAIN 
    tMax = 10 #4.                               # (ps) bound for time mesh 
    Nt =  2048                              # (-) number of sample points: t-axis
    zMax = 10 #1.                               # (m) upper limit for propagation routine
    Nz = 20000                              # (-) number of sample points: z-axis
    nSkip = 100                             # (-) number of z-steps to keep 

    # -- INITIALIZE COMPUTATIONAL DOMAIN
    t = np.linspace(-tMax, tMax, Nt, endpoint=False)
    _z = np.linspace(0, zMax, Nz, endpoint=True)

    # -- DEFINE INTIAL REAL-VALUED FIELD. INITIALLY THE PULSE AMPLITUDE IS SET
    # TO A VALUE THAT YIELDS A FUNDAMENTAL NSE SOLITON
    A0 = np.sqrt(P0)/np.cosh(t/t0)

    z, Azt = SSFM_HONSE_symmetric(_z, t, A0, beta2, beta3, beta4, gamma, s, nSkip)

    # -- POSTPROCESS RESULTS
    figure_pulse_shape(z, t, Azt, tLim = (-6.5,6.5))

if __name__ == "__main__":
    main_a()

