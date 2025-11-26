
import numpy as np
import matplotlib.pyplot as plt
from figures import figure_2a
from split_step_solver import SSFM_HONSE_symmetric

def main_a(): 

    # -- SET FIBER PARAMETERS=
    beta2 = -1 #-0.01276                        # (ps^2/m)
    beta3 = 0 #8.119*1e-5                     # (ps^3/m)
    beta4 = 0 #-1.321*1e-7                     # (ps^4/m)
    gamma = 1 #0.045                           # (1/W/m) nonlinear coefficient  
    s = 0.2                                    # self-steepening parameter     
    # -- SET PULSE PARAMETERS
    t0 = 1 #0.0284                             # (ps) pulse duration
    P0 = np.abs(beta2)/t0/t0/gamma             # (W) pulse peak power
    # -- SET PARAMETERS FOR COMPUTATIONAL DOMAIN 
    tMax = 50                                  # (ps) bound for time mesh 
    Nt =  2048                                 # (-) number of sample points: t-axis
    zMax = 12                                  # (m) upper limit for propagation routine
    Nz = 20000                                 # (-) number of sample points: z-axis
    nSkip = 100                                # (-) number of z-steps to keep 
    
    # -- INITIALIZE COMPUTATIONAL DOMAIN
    t = np.linspace(-tMax, tMax, Nt, endpoint=False)
    _z = np.linspace(0, zMax, Nz, endpoint=True)

    # -- DEFINE INTIAL REAL-VALUED FIELD. INITIALLY THE PULSE AMPLITUDE IS SET
    # TO A VALUE THAT YIELDS A FUNDAMENTAL NSE SOLITON
    A0 = np.sqrt(P0)/np.cosh(t/t0)

    # -- PROPAGATE
    z, Azt = SSFM_HONSE_symmetric(_z, t, A0, beta2, beta3, beta4, gamma, s , nSkip)

    # -- POSTPROCESS RESULTS: draw using figure_2a, then overlay guide line
    figure_2a(z, t, Azt, s, tLim=(-12,12), wLim=(-10,10), oName="figure_45a.png")
    
    #Lambda = (1550 * 10**(-9))
    #omega = 2 * np.pi * 3 * 10**8 / Lambda
    #I = np.max(np.abs(A0))**2

    #Vg_inv = gamma * I / omega
    
    #print(Vg_inv)
    

if __name__ == "__main__":
    main_a()

# EOF: main_ex04_part2_ID.py
