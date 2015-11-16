import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import matplotlib.pyplot as plt


__all__ = ['solve_opt_ric',
           'solve_fbft',
           'solve_algric',
           'solve_cl_sys',
           'comp_firstorder_mats',
           'get_forwarddiff',
           'ltvggl_bwprobmats',
           'ltvggl_fwdprobnmats',
           'linoptsys_ltvgglholo']


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
                 tmesh=None, fbd=None, ftd=None, zini=None, retaslist=False):
    """solve the (closed loop) forward problem

    by the implicit trapezoidal rule

    Parameters
    ---
    retaslist : boolean, optional
        whether to return the output as a list, defaults to `False`

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
    if retaslist:
        inplist = [inpdict[t][0][0]]
        outplist = [outdict[t][0][0]]

    for tk, t in enumerate(tmesh[1:]):
        cts = t - tmesh[tk]
        inpdict.update({t: _feedthrough(t) +
                       np.dot(_feedbackgain(t), zc)})
        crhs = zc + 0.5*cts*(2*f + np.dot(A + _feedbackgain(t-cts), zc) +
                             _feedthrough(t) + _feedthrough(t-cts))
        cmat = M - 0.5*cts*(A + _feedbackgain(t))
        zc = np.linalg.solve(cmat, crhs)
        outdict.update({t: np.dot(C, zc)})
        if retaslist:
            inplist.append(inpdict[t][0][0])
            outplist.append(outdict[t][0][0])
    if retaslist:
        return outplist, inplist
    else:
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


def get_forwarddiff(tmesh):
    try:
        N = tmesh.size
    except AttributeError:
        N = len(tmesh)
    # define the discrete difference matrix
    try:
        hvec = tmesh[1:] - tmesh[:-1]
    except TypeError:
        hvec = np.atleast_2d(np.array(tmesh[1:]) - np.array(tmesh[:-1])).T
    bigeye = np.eye(N, N)
    return (bigeye[1:, :] - bigeye[:-1, :])/hvec


def _zspm(nl, nc):
    return sps.csc_matrix((nl, nc))


def ltvggl_bwprobmats(tmesh=None, mmat=None, getgmat=None,
                      getdgmat=None, getamat=None, vold=None,
                      l1term=None, l2term=None, dualrhs=None,
                      grhs=None, dgrhs=None, nr=None):
    """ get the coefficient matrix for the backward/adjoint problem

    -M.T*l1' = A.T*l2 - G.Tm1 - dG.Tm2
    -M.T*l2' = M.T*l1 - G.Tm2
    0 = Gl1
    0 = dGl1 + Gl2

    dims        ntp-1*nx nx   ntp-1*nx   nx     ntp-1*nr ntp-1*nr
               ------------------------------------------------------
             |
    nx       |  0        tl1m   0      0       0       0
    nx       |  0        0      0      tl2m    0       0
    ntp-1*nx |   -diff_M        -A.T   0       G.T     dG.T
    ntp-1*nx |  -M.T     0       -diff_M       0       G.T
    ntp-1*nr |  G        0      0      0       0       0
    ntp-1*nr |  dG       0      G      0       0       0

    *

    coeffvec
    [l1     tl1     l2  tl2     m1  m2].T

    =

    [QC     0   0   0   0   0].T*ystar
    """

    nx = l1term.size
    ntp = len(tmesh)
    bwintmesh = tmesh[:-1]
    ntpi = len(bwintmesh)

    diffmat = get_forwarddiff(tmesh)
    spdfm = sps.csc_matrix(-diffmat)
    diffm = sps.kron(spdfm, mmat.T)
    tdmmat = sps.kron(sps.eye(ntpi), mmat.T)

    adiag, gdiag, dgdiag = [], [], []
    rhsl = []

    for curt in bwintmesh:
        adiag.append(sps.csc_matrix(getamat(curt)).T)
        gdiag.append(getgmat(curt))
        dgdiag.append(getdgmat(curt))
        rhsl.append(dualrhs(curt))

    tdamatt = sps.block_diag(adiag)
    tdgmat = sps.block_diag(gdiag)
    dtdgmat = sps.block_diag(dgdiag)
    lrhs = np.hstack(rhsl).reshape((ntpi*nx, 1))

    tl1mat = sps.hstack([_zspm(nx, ntpi*nx), sps.eye(nx),
                         _zspm(nx, ntp*nx+2*ntpi*nr)])
    tl2mat = sps.hstack([_zspm(nx, (2*ntp-1)*nx), sps.eye(nx),
                         _zspm(nx, 2*ntpi*nr)])
    coefmatl1 = sps.hstack([-diffm, -tdamatt, _zspm(ntpi*nx, nx),
                            tdgmat.T, dtdgmat.T])
    coefmatl2 = sps.hstack([-tdmmat, _zspm(ntpi*nx, nx), -diffm,
                            _zspm(ntpi*nx, ntpi*nr), tdgmat.T])
    coefmatg = sps.hstack([tdgmat, _zspm(ntpi*nr, nx),
                           _zspm(ntpi*nr, nx+ntpi*(nx+2*nr))])
    coefmatdg = sps.hstack([dtdgmat, _zspm(ntpi*nr, nx), _zspm(ntpi*nr, nx),
                            tdgmat, _zspm(ntpi*nr, ntpi*2*nr)])

    coefmat = sps.vstack([tl2mat, tl1mat, coefmatl1, coefmatl2,
                          coefmatg, coefmatdg]).tocsc()
    rhs = np.vstack([l1term, l2term, lrhs, np.zeros((ntpi*(nx+2*nr), 1))])
    return coefmat, rhs


def ltvggl_fwdprobnmats(tmesh=None, mmat=None, bmat=None, inpufun=None,
                        getgmat=None, getdgmat=None, getamat=None, vold=None,
                        xini=None, vini=None,
                        xrhs=None, grhs=None, dgrhs=None, nr=None,
                        onlyretmats=False):
    ''' model structure
    Mx' = Mv - G.Tq - DG.Tp
    Mv' = Ax - G.Tp + Bu + rhs
       g = Gx
      dg = DGx + Gv
    with A, G, rhs time-varying
    '''

    nx = xini.size
    ntp = len(tmesh)
    intmesh = tmesh[1:]
    ntpi = len(intmesh)
    tsiinv = 1./(tmesh[1] - tmesh[0])

    diffmat = get_forwarddiff(tmesh)
    spdfm = sps.csc_matrix(diffmat)
    diffm = sps.kron(spdfm, mmat)
    tdmmat = sps.kron(sps.eye(ntpi), mmat)

    adiag, gdiag, dgdiag = [], [], []
    xrhsl, grhsl, dgrhsl = [], [], []

    def fwdrhs(t):
        if bmat is None or inpufun is None:
            return xrhs(t).reshape((nx, ))
        else:
            return (bmat.dot(inpufun(curt)) + xrhs(curt)).reshape((nx, ))

    for curt in intmesh:
        adiag.append(sps.csc_matrix(getamat(curt)))
        gdiag.append(getgmat(curt))
        dgdiag.append(getdgmat(curt))
        xrhsl.append(fwdrhs(curt))
        grhsl.append(grhs(curt))
        dgrhsl.append(dgrhs(curt))

    tdamat = sps.block_diag(adiag)

    tdgmat = sps.block_diag(gdiag)
    dtdgmat = sps.block_diag(dgdiag)
    tdrhs = np.hstack([xrhsl]).reshape((ntpi*nx, 1))
    grhs = np.hstack([grhsl]).reshape((ntpi*nr, 1))
    dgrhs = np.hstack([dgrhsl]).reshape((ntpi*nr, 1))

    """ what the coeffmat will look like:

    dims        nx   ntp-1*nx   nx      ntp-1*nx    ntp-1*nr ntp-1*nr
               -------------------------
             |
    nx       |  inixm   0       0       0           0       0
    nx       |  0       0       iniv    0           0       0
    ntp-1*nx |  --diff_M--      0       -M          G.T     dG.T
    ntp-1*nx |  0       -A      --diff_M--          0       G.T
    ntp-1*nr |  0       G       0       0           0       0
    ntp-1*nr |  0       dG      0       G           0       0

    *

    coeffvec
    [x0     x   v0  v   q   p].T

    =

    [inix   iniv    0   xrhs    -g(xold)    -G*vold]

    where xres = -diff_M*vold + Ax - G.Tp +Bu

    as it comes from the Newton iteration
    """

    inixmat = sps.hstack([sps.eye(nx), _zspm(nx, (2*ntp-1)*nx+2*ntpi*nr)])
    inivmat = tsiinv*sps.hstack([_zspm(nx, ntp*nx), sps.eye(nx),
                                 _zspm(nx, ntpi*(nx+2*nr))])
    coefmatdx = sps.hstack([diffm, _zspm(ntpi*nx, nx), -tdmmat,
                            tdgmat.T, dtdgmat.T])
    coefmatdv = sps.hstack([_zspm(ntpi*nx, nx), -tdamat, diffm,
                            _zspm(ntpi*nx, ntpi*nr), tdgmat.T])
    coefmatg = sps.hstack([_zspm(ntpi*nr, nx), tdgmat,
                           _zspm(ntpi*nr, nx+ntpi*(nx+2*nr))])
    coefmatdg = sps.hstack([_zspm(ntpi*nr, nx), dtdgmat, _zspm(ntpi*nr, nx),
                            tdgmat, _zspm(ntpi*nr, ntpi*2*nr)])

    coefmat = sps.vstack([inixmat, inivmat, coefmatdx,
                          coefmatdv, coefmatg, coefmatdg]).tocsc()
    rhsx = np.vstack([xini, vini, np.zeros((ntpi*nx, 1)), tdrhs,
                      grhs, dgrhs])

    if onlyretmats:
        return coefmat, xrhs
    else:
        tdsol = spsla.spsolve(coefmat, rhsx)
        return tdsol


def linoptsys_ltvgglholo(tmesh=None, mmat=None, bmat=None, inpufun=None,
                         getgmat=None, getdgmat=None, getamat=None, vold=None,
                         xini=None, vini=None, qmat=None, smat=None,
                         rmatinv=None, cmat=None, ystar=None,
                         xrhs=None, grhs=None, dgrhs=None, nr=None):
    '''
    Solve the coupled system of

    Mx' = Mv - G.Tq - DG.Tp
    Mv' = Ax - G.Tp + rhs + BR.-1B.T*l2
       g = Gx
      dg = DGx + Gv

    x(0) = inix, v(0) = iniv

    and its adjoint

    -M.T*l1' = A.T*l2 - G.Tm1 - dG.Tm2 - C.TQCx + C.TQy
    -M.T*l2' = M.T*l1 - G.Tm2
    0 = Gl1
    0 = dGl1 + Gl2

    l1(tE) = 0, l2(tE) = -C.T*S*(Cx(tE) - y(tE))
    '''
    biga, fwdrhs = \
        ltvggl_fwdprobnmats(tmesh=tmesh, mmat=mmat, getgmat=getgmat,
                            getdgmat=getdgmat, getamat=getamat, vold=vold,
                            xini=xini, vini=vini, xrhs=xrhs, grhs=grhs,
                            dgrhs=dgrhs, nr=nr, onlyretmats=True)

    def dualrhs(t):
        return np.dot(cmat.T, np.dot(qmat, ystar(tE)))
    tE = tmesh[-1]
    l1term = 0*xini
    l2term = np.dot(cmat.T, np.dot(smat, ystar(tE)))
    bigat, dualrhs = \
        ltvggl_bwprobmats(tmesh=tmesh, mmat=mmat, getgmat=getgmat,
                          getdgmat=getdgmat, getamat=getamat, vold=vold,
                          l1term=l1term, l2term=l2term, dualrhs=dualrhs,
                          grhs=grhs, dgrhs=dgrhs, nr=nr)
    nx = xini.size
    ntp = len(tmesh)
    ntpi = ntp-1
    brmibt = np.dot(bmat, np.dot(rmatinv, bmat.T))
    tdbrmibt = sps.kron(sps.eye(ntpi), brmibt)
    bigbrmbtx = sps.hstack([_zspm(ntpi*nx, (ntp+1)*nx), tdbrmibt,
                            _zspm(ntpi*nx, 2*ntpi*nr)])
    bigbrmbt = sps.vstack([_zspm((2+ntpi)*nx, 2*(ntp*nx + ntpi*nr)),
                           bigbrmbtx])
    ctqc = np.dot(cmat.T, np.dot(qmat, cmat))
    tdctqc = sps.kron(sps.eye(ntpi), ctqc)
    tl1ctscm = sps.hstack([_zspm(nx, ntpi*nx), np.dot(cmat.T, smat.dot(cmat)),
                           _zspm(nx, ntp*nx+2*ntpi*nr)])
    bigctqc = sps.vstack([tl1ctscm, _zspm(nx, 2*(ntp*nx+ntpi*nr)),
                          tdctqc, _zspm(ntpi*nx+2*ntp*nr, 2*(ntp*nx+ntpi*nr))])
    bigcfm = sps.vstack([sps.hstack([biga, -bigbrmbt]),
                         sps.hstack([bigctqc, bigat])])
    bigrhs = np.vstack([xrhs, dualrhs])

    xvqpllmm = spsla.spsolve(bigcfm, bigrhs)

    return xvqpllmm
