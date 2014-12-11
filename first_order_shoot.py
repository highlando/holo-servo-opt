import numpy as np
import scipy.integrate as spin
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
#    h = np.vstack([ fpri, fdua, np.zeros((2, 1)) ])   			# gemischt funktion und vektor!

    # boundary conditions:  [B01; 0] y(0) + [0; BT2] y(T) = [init1; init2]
      # FEHLT!
     
     
    h = np.zeros((2*NN+2, 1))
    y0 = np.ones((2*NN+2, 1))
    
    print 'starte Solver'
    t0 = 0
    T = 1
    tau = 0.1
        
    y = solveODE(M, K, h, y0=y0, t0=t0, T=T, tau=tau)
    
    print 2*NN+2
    print y.shape
    
   
      
    ### 
    # Step 1: find fundamental system (brauche nur Endwert Y(T))
    Y_init = np.vstack([ np.hstack([np.eye(NN), np.zeros((NN, 1))]),
                         np.zeros((NN, NN+1)),
                         np.hstack([np.zeros((1, NN)), np.ones((1,1))]), 
                         np.zeros((1, NN+1)) ])   # size(Y) = 2NN+2, needed ic = NN+1
#    Y_end = solveODE(M, K, 0, Y_init)


    ### 
    # Step 2: find solution of inhom system (brauche nur Endwert w(T))
    w_init = np.vstack([ np.zeros((NN,1)), zini, np.zeros((2,1)) ])
#    w_end = solveODE(M, K, h, w_init)
    

    ### 
    # Step 3: find coefficient vector s
    rhs = init2 - BT2*w_end
    matrix = BT2*Y_end    
    # matrix*s = rhs

    ### 
    # Step 4: get initial condition for y and solve IVP
    y0 = Y_init*s + w_init    
  
    
    # print '\nf_fo=\n', fpri(0)     


    return 2


# solve an IVP of the form   My' = Ky + h
# 
def solveODE(M, K, h, y0, t0, T, tau):
    print 'in solve'
    # write My' = Ky + h in the form y' = f(y)
    # thus f(y) = z where z is the solution of Mz = b := Ky + h     
    def func(t, y):
        #return np.array( spla.solve(M, K*y + h) )
        return np.ones(y.shape)
            
    # t = time point in which ODE should be solved
    # h0, hmax, hmin = first, maximal, minimal step size
    # mxstep = maximal number of steps
    # mxordn, mxords = maximal order for nonsitff(Adams)/stiff(BDF) problems
 #   y = spin.odeint(func=func, y0=init, t=init, mxstep=1000, mxordn=2, mxords=2)    
         
    r = spin.ode(func)
    r.set_integrator('vode', method='bdf', with_jacobian=False)
    r.set_initial_value(y0, t0)     
        
    print 'enter integration loooop'
    while r.successful() and r.t < T:        
        r.integrate(r.t+tau)      

    return r.y




    
