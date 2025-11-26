"""main_ex04_part2_ID.py

Computer Exercises for Lecture "Computational Photonics"

Exercise 04 -  The nonlinear Schroedinger equation - Part II

NOTE: Before you start working on the assignments and before you hand in your
solution, carefully read the instructions below.

    * You can work on these assignments in groups of up to 3 students. Note,
      however, that everybody has to hand in his/her own solution.

    * Please personalize your completed script main_ex03_part2 ID.py by replacing
      the handle ID by your student-id. For redundancy, make sure to also
      include your name and student-id in the completed scripts that you hand
      in. We therefore provided three metadata variables
          __STUDENT_NAME__ = "YOUR NAME HERE"
          __STUDENT_ID__   = "YOUR STUDENT ID HERE"
          __GROUP_MEMBERS__ = "IF YOU WORKED AS PART OF A GROUP, LIST YOUR GROUP MEMBERS HERE"
      at the beginning of the script.

    * To complete the code, look out for TODO statements, they will provide
      further hints!

DATE: 
    2021-06-01
    2023-05-16 (minor edits)
    2024-05-24 (minor edits)
    2025-05-23 (minor edits)
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
    #ax.set_xticks([-6,-3,0,3,6])
    #ax.set_xlabel("Time $t~(\mathrm{ps})$")   
    ax.set_xlabel("Time $t$")
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Intensity $I$")

    ax.legend(
        title="$z/L_{D}$",    
        loc = "upper left"
        #loc = (0.3,0.7)
    )
    
    plt.show()

def main_a():

    # -- SET FIBER PARAMETERS
    beta2 = -1 #-0.01276                        # (ps^2/m)
    beta3 = 0 #8.119*1e-5                     # (ps^3/m)
    beta4 = 0 #-1.321*1e-7                     # (ps^4/m)
    gamma = 1 #0.045                           # (1/W/m) nonlinear coefficient
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

    # TODO: THE FUNCTION BELOW IS NOT IMPLEMENTE, YET. FOLLOW THE STEPS ON THE
    # WORKSHEET TO COMPLETE THE CODE
    z, Azt = SSFM_HONSE_symmetric(_z, t, A0, beta2, beta3, beta4, gamma, nSkip)

    # -- POSTPROCESS RESULTS
    figure_pulse_shape(z, t, Azt, tLim = (-6.5,6.5))
    
    # The intensity vs time graph for difference values of z/L0
    # for 0, 5, and 10. 
    #
    #

if __name__ == "__main__":
    main_a()

# EOF: main_ex04_part2_ID.py
