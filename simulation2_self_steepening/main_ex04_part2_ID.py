import numpy as np
from figures import figure_2a
from split_step_solver import SSFM_HONSE_symmetric

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
    tMax = 50 #4.                               # (ps) bound for time mesh 
    Nt =  2048                              # (-) number of sample points: t-axis
    zMax = 12 #1.                               # (m) upper limit for propagation routine
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
    figure_2a(z, t, Azt, tLim = (-12,12), wLim = (-10,10), oName="Figure_2.png")

    ANSWER_b1 = "YOUR ANSWER HERE"
    ANSWER_b2 = "YOUR ANSWER HERE"


if __name__ == "__main__":
    main_a()
