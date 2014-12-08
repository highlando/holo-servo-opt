import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt


def solve_opt_shoot(tA=None, beta0=None, beta1=None, tB=None, tC=None, gamma=None,
                  fpri=None, fdua=None, zini=None, bt=None, tmesh=None):

    # first order system (also in input)
    NN = tA.shape[0]
    # B: (N, 1) array,  tB: (2N, 1) array,  ttB: (2N, 2) array 
    zerov = np.zeros((NN, 1))
    
    ttB = np.hstack([tB, zerov])
    tJ = np.vstack([np.array([0, beta1]), np.array([-beta1, 0])])
    tI = np.vstack([np.array([-beta0, 0]), np.array([0, beta1])])
    
    # first-order system in the form:   My' = Ky + h
    M = np.vstack([ np.hstack([np.zeros((NN, NN)), np.eye(NN), np.zeros((NN, 2))]),
                    np.hstack([-np.eye(NN), np.zeros((NN, NN)), np.zeros((NN, 2))]),
                    np.hstack([np.zeros((2, 2*NN)), tJ]) ])
    K = np.vstack([ np.hstack([np.zeros((NN, NN)), tA, ttB]),
                    np.hstack([tA.T, -tC.T*tC, np.zeros((NN, 2))]),
                    np.hstack([ttB.T, np.zeros((2, NN)), tI]) ])
    h = np.vstack([ fpri, fdua, np.zeros((2, 1)) ])   			# gemischt funktion und vektor!

    # boundary conditions:  [B01; 0] y(0) + [0; BT2] y(T) = [init1; init2]
      # FEHLT!
      
    ### 
    # Step 1: find fundamental system (brauche nur Endwert Y(T))
    Y_init = np.vstack([ np.hstack([np.eye(NN), np.zeros((NN, 1))]),
                         np.zeros((NN, NN+1)),
                         np.hstack([np.zeros((1, NN)), 1]), 
                         np.zeros((1, NN+1)) ])   # size(Y) = 2NN+2, needed ic = NN+1
    Y_end = solveODE(M, K, 0, mesh, Y_init)


    ### 
    # Step 2: find solution of inhom system (brauche nur Endwert w(T))
    w_init = np.vstack([ np.zeros((NN,1)), zini, np.zeros((2,1)) ])
    w_end = solveODE(M, K, h, mesh, w_init)
    

    ### 
    # Step 3: find coefficient vector s
    rhs = init2 - BT2*w_end
    matrix = BT2*Y_end    
    # matrix*s = rhs

    ### 
    # Step 4: get initial condition for y and solve IVP
    y_init = Y_init*s + w_init
    

    
    
    # print '\nf_fo=\n', fpri(0) 
    


    return 2


# solve an IVP of the form   My' = Ky + h
# 
def solveODE(M, K, h, mesh, init):


    return 




    
