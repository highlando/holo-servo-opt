import numpy as np


def get_abcf(kvec=None, mvec=None, dvec=None, N=None, printmats=False):
    """system matrices for the multi mass servo problem

    `F -> mN <-> mN-1 <-> .... <-> m1`

    Parameters
    ----------
    kvec : (N-1, ) array, optional
       the spring constants
    mvec : (N, ) array, optional
       the masses
    dvec : (N, ) array, optional
        the spring offsets
    N : integer, optional
        if provided, the matrices for a system of size `N`
        with all parameters equal to `1` is returned
    printmats : boolean, optional
        whether to print the matrices to the screen, defaults to false

    Returns
    -------
    A : (N, N) array
        coefficient matrix like `kvec/mvec*delta_x`
    B : (N, 1) array
        `(0, 0, ..., 1/mvec[-1]).T`
    C : (1, N) array
        `(1, 0, ..., 0)`
    f : (N, 1) array
        inhomogeneity, basically `kvec/mvec*dvec`
    """
    if N is None:
        N = mvec.size
    else:
        kvec = np.ones((N-1, 1))
        dvec = np.ones((N-1, 1))
        mvec = np.ones((N, 1))

    A = np.zeros((N, N))
    f = np.zeros((N, 1))
    B = np.zeros((N, 1))
    C = np.zeros((1, N))

    difftonext = np.array([[1, -1]])

    # eqn for first mass body m1
    k = 0
    A[k, k:k+2] = -kvec[k]/mvec[k]*difftonext
    C[0, k] = 1
    f[k] = kvec[k]/mvec[k]*dvec[k]

    # eqns for the inner mass bodies
    for k in range(1, N-1):
        # consider the next body
        A[k, k:k+2] = -kvec[k]/mvec[k]*difftonext
        f[k] += kvec[k]/mvec[k]*dvec[k]
        # consider the previous body
        A[k, k-1: k+1] = A[k, k-1: k+1] + kvec[k-1]/mvec[k]*difftonext
        f[k] += -kvec[k-1]/mvec[k]*dvec[k-1]

    # eqn for last mass body
    k = N-1
    A[k, k-1: k+1] = kvec[k-1]/mvec[k]*difftonext
    f[k] = -kvec[k-1]/mvec[k]*dvec[k-1]
    B[k, 0] = 1./mvec[-1]
    if printmats:
        print '\nA=\n', A
        print '\nB=\n', B
        print '\nC=\n', C
        print '\nf=\n', f

    return A, B, C, f


def get_trajec(trgt, tE=None, g0=None, gf=None,
               trnsarea=None, tanpa=None, tM=None,
               polydeg=None):
    if trgt == 'pwl':
        tM = tE/2

        def trajec(t):
            if t < tM - trnsarea/2:
                g = g0
            elif t > tM + trnsarea/2:
                g = gf
            else:
                g = g0 + (gf-g0)*(t - tM + trnsarea/2)/trnsarea
            return g

    elif trgt == 'pwp':
        tM = tE/2

        def trajec(t):
            if t < tM - trnsarea/2:
                g = g0
            elif t > tM + trnsarea/2:
                g = gf
            else:
                s = (t + -tM + trnsarea/2)/trnsarea
                if polydeg == 1:
                    g = g0 + (gf-g0)*s
                elif polydeg == 3:
                    g = g0 + (gf-g0)*(3*s**2 - 2*s**3)
                elif polydeg == 5:
                    g = g0 + (gf-g0)*(10*s**3 - 15*s**4 + 6*s**5)
                elif polydeg == 7:
                    g = g0 + (gf-g0)*(35*s**4 - 84*s**5 + 70*s**6 - 20*s**7)
                elif polydeg == 9:
                    g = g0 + (126*s**5 - 420*s**6 + 540*s**7 -
                              315*s**8 + 70*s**9)*(gf - g0)
                else:
                    raise Warning('polydeg needs be defined')
            return g

    elif trgt == 'atan':
        tM = tE/2

        def trajec(t):
            return 1 + g0 + np.tanh(tanpa*(t - tM))

    elif trgt == 'plnm':

        def trajec(t):
            trt = t/tE
            return g0 + (126*trt**5 - 420*trt**6 + 540*trt**7 -
                         315*trt**8 + 70*trt**9)*(gf - g0)

    return trajec


def get_tint(t0, tE, Nts, sqzmesh=True, plotmesh=False):
    """set up a (nonuniform) time mesh

    for tracking problems a mesh with nodes clustered at the ends
    is advisable since

     1. the backward in time Riccati problem needs a high resolution towards \
    the terminal value
     2. the closed loop problem has large gradients at the beginning

    this function uses the sin function to squeeze a mesh towards the \
    marginal points
    """
    if sqzmesh:
        taux = np.linspace(-0.5*np.pi, 0.5*np.pi, Nts+1)
        taux = (np.sin(taux) + 1)*0.5  # squeeze and adjust to [0, 1]
        tint = (t0 + (tE-t0)*taux).flatten()  # adjust to [t0, tE]
    else:
        tint = np.linspace(t0, tE, Nts+1).flatten()
    if plotmesh:
        import matplotlib.pyplot as plt
        plt.plot(tint, np.ones(Nts+1), '.')
        plt.show()

    return tint
