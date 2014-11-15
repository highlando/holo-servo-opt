import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt


def solve_cl_sys(A=None, B=None, C=None, bmo=None, f=None,
                 tmesh=None, fbd=None, ftd=None, zini=None):
    """solve the (closed loop) forward problem

    """
    t, zc = tmesh[0], zini
    M = np.eye(A.shape[0])
    outdict = {t: np.dot(C, zc)}
    for tk, t in enumerate(tmesh[1:]):
        cts = t - tmesh[tk]
        crhs = zc + cts*(f + np.dot(B, bmo*ftd[t]))
        cmat = M - cts*(A + np.dot(B, bmo*fbd[t]))
        zc = np.linalg.solve(cmat, crhs)
        outdict.update({t: np.dot(C, zc)})
    return outdict


def plot_output(tmesh, outdict, targetsig=None):
    plt.figure(44)
    outsigl = []
    trgsigl = []
    for t in tmesh:
        outsigl.append(outdict[t][0][0])
        trgsigl.append(targetsig(t))

    plt.plot(tmesh, outsigl)
    plt.plot(tmesh, trgsigl)


def plot_fbft(tmesh, fbdict, ftdict):
    normfbl = []
    ftl = []
    for t in tmesh:
        normfbl.append(np.linalg.norm(fbdict[t]))
        ftl.append(ftdict[t][0])

    plt.figure(11)
    plt.plot(tmesh, normfbl)
    plt.figure(22)
    plt.plot(tmesh, ftl)


def solve_fbft(A=None, bbt=None, ctc=None, fpri=None, fdua=None,
               tmesh=None, termx=None, termw=None, bt=None, verbose=False):
    """solve for feedback gain and feedthrough

    by solving a differential Riccati equation and a backward problem \
    for the feedthrough.

    By now, we simply use backward Euler

    Parameter
    ---------
    fpri : f(t), callable
        the value of the rhs in the primal eqns at time `t`
    fdua : f(t), callable
        the value of the rhs in the dual eqns at time `t`

    Returns
    -------
    fbdict : dict
        with time `t` as key and `B.T*X(t)` as value
    ftdict : dict
        with time `t` as key and the feedthrough `B.T*w(t)` as value
    """

    t = tmesh[-1]
    fbdict = {t: np.dot(bt, termx)}
    ftdict = {t: np.dot(bt, termw)}

    xnorml = [np.linalg.norm(termx)]

    Xc = termx
    wc = termw

    M = np.eye(A.shape[0])

    for tk, t in reversed(list(enumerate(tmesh[:-1]))):
        cts = tmesh[tk+1] - t
        if verbose:
            print 'Time is {0}, timestep is {1}'.format(t, cts)

        # integrating the Riccati Equation
        fmat = -0.5*M + cts*A
        W = -Xc + cts*ctc
        Xp = solve_algric(A=fmat, W=W, R=cts*bbt, X0=Xc)
        fbdict.update({t: np.dot(bt, Xp)})

        # timestepping for the feedthrough variable
        prhs = wc + cts*np.dot(Xp, fpri(t)) + cts*fdua(t)
        wp = np.linalg.solve(M - cts*(A.T + np.dot(Xp, bbt)), prhs)
        ftdict.update({t: np.dot(bt, wp)})
        xnorml.append(np.linalg.norm(Xp))
        Xc = Xp
        wc = wp

    plotxnorm = True
    if plotxnorm:
        plt.figure(33)
        plt.plot(tmesh, xnorml)

    return fbdict, ftdict


def solve_algric(A=None, R=None, W=None, X0=None,
                 nwtnstps=20, nwtntol=1e-12,
                 verbose=False):
    """ solve algebraic Riccati Equation via Newton iteration

    for dense matrices.

    The considered Ric Eqn is `A.TX + XA + XRX = W`
    """
    XN = X0
    nwtns, nwtnres = 0, 999
    while nwtns < nwtnstps and nwtnres > nwtntol:
        nwtns += 1
        fmat = A + np.dot(R, XN)
        cw = W + np.dot(XN, np.dot(R, XN))
        XNN = spla.solve_lyapunov(fmat.T, cw)
        nwtnres = np.linalg.norm(XN - XNN)
        XN = XNN
        if verbose:
            print 'Newton step {0}: norm of update {1}'.format(nwtns, nwtnres)
    if verbose:
        ricres = np.dot(A.T, XN) + np.dot(XN, A) + \
            np.dot(XN, np.dot(R, XN)) - W
        print 'Ric Residual: {0}'.format(np.linalg.norm(ricres))
    return XN


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


def comp_firstorder_mats(A=None, B=None, C=None, f=None,
                         posini=None, velini=None,
                         damp=False):
    """compute the matrices for the reformulation as a first order system

    Returns
    -------
    tA : (2N, 2N) array
    tB : (2N, 1) array
    tC : (1, 2N) array
    tf : (2N, 1) array
    tini : (2N, 1) array
        the initial value `[posini; velini]`
    """
    N = A.shape[0]
    tA = np.vstack([np.hstack([np.zeros((N, N)), np.eye(N)]),
                    np.hstack([A, np.zeros((N, N))])])
    zerov = np.zeros((N, 1))
    tB = np.vstack([zerov, B])
    tC = np.hstack([C, zerov.T])
    # try vel observation
    # tC = np.hstack([zerov.T, C])

    tf = np.vstack([zerov, f])

    tini = np.vstack([posini, velini])

    return tA, tB, tC, tf, tini


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
        `(0, 0, ..., 1).T`
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
    B[k, 0] = 1
    if printmats:
        print '\nA=\n', A
        print '\nB=\n', B
        print '\nC=\n', C
        print '\nf=\n', f

    return A, B, C, f

if __name__ == '__main__':
    defsysdict = dict(mvec=np.array([2., 1.]),
                      dvec=np.array([0.5]),
                      kvec=np.array([10.]),
                      printmats=True)
    defprbdict = dict(posini=np.array([[0.5], [0]]),
                      velini=np.array([[0.], [0.]]))
    tE = 6.
    Nts = 599
    g0, gf = 0.5, 2.5

    def trajec(t):
        trt = t/tE
        return g0 + (126*trt**5 - 420*trt**6 + 540*trt**7 -
                     315*trt**8 + 70*trt**9)*(gf - g0)

    defctrldict = dict(gamma=1e-3,
                       beta=1e-3,
                       g=trajec)
    A, B, C, f = get_abcf(**defsysdict)
    tA, tB, tC, tf, tini = comp_firstorder_mats(A=A, B=B, C=C, f=f,
                                                **defprbdict)
    tmesh = get_tint(0.0, tE, Nts, sqzmesh=False, plotmesh=False)

    beta, gamma = defctrldict['beta'], defctrldict['gamma']
    bbt = 1./beta*np.dot(tB, tB.T)
    ctc = np.dot(tC.T, tC)

    def fpri(t):
        return tf

    def fdua(t):
        return np.dot(tC.T, trajec(t))

    termw = gamma*fdua(tmesh[-1])
    termx = -gamma*ctc

    fbdict, ftdict = solve_fbft(A=tA, bbt=bbt, ctc=ctc, fpri=fpri, fdua=fdua,
                                tmesh=tmesh, termx=termx, termw=termw, bt=tB.T)

    sysout = solve_cl_sys(A=tA, B=tB, C=tC, bmo=1./beta, f=tf,
                          tmesh=tmesh, fbd=fbdict, ftd=ftdict, zini=tini)

    plot_output(tmesh, sysout, trajec)
    plot_fbft(tmesh, fbdict, ftdict)
    plt.show(block=False)
