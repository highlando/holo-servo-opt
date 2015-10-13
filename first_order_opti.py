import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt


def solve_opt_ric(A=None, beta=None, B=None, C=None, gamma=None,
                  fpri=None, fdua=None, bt=None, tmesh=None):
    bbt = 1./beta*np.dot(B, B.T)
    ctc = np.dot(C.T, C)
    termw = gamma*fdua(tmesh[-1])
    termx = -gamma*ctc

    fbdict, ftdict = solve_fbft(A=A, bbt=bbt, ctc=ctc,
                                fpri=fpri, fdua=fdua, tmesh=tmesh,
                                termx=termx, termw=termw, bt=bt)

    return fbdict, ftdict


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

    plotxnorm = False
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


def solve_cl_sys(A=None, B=None, C=None, bmo=None, f=None,
                 tmesh=None, fbd=None, ftd=None, zini=None):
    """solve the (closed loop) forward problem

    by implicit Euler
    """
    t, zc = tmesh[0], zini
    M = np.eye(A.shape[0])

    def _feedbackgain(t):
        if fbd is None:
            return 0*A  # a zero matrix
        else:
            return np.dot(B, bmo*fbd[t])

    def _feedthrough(t):
        if ftd is None:
            return np.zeros((B.shape[0], 1))  # a zero input
        else:
            if ftd.__class__ == dict:
                return np.dot(B, bmo*ftd[t])
            else:
                try:
                    return np.dot(B, bmo*ftd(t))  # maybe it is a function
                except TypeError:
                    return np.dot(B, bmo*ftd)  # or a constant??

    outdict = {t: np.dot(C, zc)}
    inpdict = {t: _feedthrough(t) + np.dot(_feedbackgain(t), zc)}

    for tk, t in enumerate(tmesh[1:]):
        cts = t - tmesh[tk]
        inpdict.update({t: _feedthrough(t) +
                       np.dot(_feedbackgain(t), zc)})
        crhs = zc + 0.5*cts*(2*f - np.dot(A, zc) +
                             _feedthrough(t) + _feedthrough(t-cts))
        cmat = M - 0.5*cts*(A + _feedbackgain(t))
        zc = np.linalg.solve(cmat, crhs)
        outdict.update({t: np.dot(C, zc)})
    return outdict, inpdict


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
