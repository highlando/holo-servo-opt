import numpy as np
import numpy.linalg as npla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla


def disc_cent_diffs(tmesh):
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
    backdiffs = (bigeye[1:-1, :] - bigeye[:-2, :])/hvec[:-1]
    forwdiffs = (bigeye[2:, :] - bigeye[1:-1, :])/hvec[1:]
    diffs = (forwdiffs - backdiffs)/hvec[1:]
    return diffs


def disc_func(f, intmesh):
    fl = []
    for t in intmesh:
        fl.append(f(t))
    return np.vstack(fl)


def fd_fullsys(A=None, B=None, C=None, flist=None, g=None,
               tmesh=None, Q=None, bone=None, bzero=None, gamma=None,
               inix=None, inidx=None, udiril=[False, False]):
    """ finite differences for the second order optimality system

    0  I  0                     -  0     A  B               = f
    I  0  0   * [l'', x'', u''] -  A.T  -Q  0   * [l, x, u] = g
    0  0 b1*I                   -  -B.T  0 b0*I             = 0
    """

    diffmat = disc_cent_diffs(tmesh)
    intmesh = tmesh[1:-1]
    nint = intmesh.size
    nx = A.shape[0]
    nu = B.shape[1]
    gvec = disc_func(g, intmesh)
    ftl = []
    for f in flist:
        ftl.append(disc_func(f, intmesh))
    fvec = np.vstack(ftl)
    rhsinner = np.vstack([fvec, np.kron(C.T, gvec), np.zeros((nint, 1))])

    xxz = np.zeros((nx*nint, nx*(nint+2)))
    xuz = np.zeros((nx*nint, nu*(nint+2)))
    uxz = np.zeros((nu*nint, nx*(nint+2)))

    xleye = np.eye(nx)
    ueye = np.eye(nu)
    diffxl = np.kron(xleye, diffmat)
    diffu = np.kron(ueye, diffmat)
    xdiff = np.hstack([xxz, diffxl, xuz])
    ldiff = np.hstack([diffxl, xxz, xuz])
    udiff = np.hstack([uxz, uxz, bone*diffu])
    bigdiff = np.vstack([xdiff, ldiff, udiff])

    eyetint = np.eye(tmesh.size)[1:-1, :]
    xxcoff = np.kron(A, eyetint)
    xucoff = np.kron(B, eyetint)
    llcoff = np.kron(A.T, eyetint)
    lxcoff = np.kron(-Q, eyetint)
    ulcoff = np.kron(-B.T, eyetint)

    uucoff = bzero*np.kron(ueye, eyetint)

    bigcoff = np.vstack([np.hstack([xxz, xxcoff, xucoff]),
                         np.hstack([llcoff, lxcoff, xuz]),
                         np.hstack([ulcoff, uxz, uucoff])])

    # the boundary conditions for [l, x, u]
    nT = tmesh.size
    # on x : x(0) = inix
    bcx = np.zeros((nx, (nx+nx+nu)*nT))
    fbcx = np.zeros((nx, 1))
    for k in range(nx):
        bcx[k, (nx+k)*nT] = 1
        fbcx[k] = inix[k]

    # on dx : dx(0) = iniv
    bcdx = np.zeros((nx, (nx+nx+nu)*nT))
    fbcdx = np.zeros((nx, 1))
    h1 = tmesh[1] - tmesh[0]
    for k in range(nx):
        bcdx[k, (nx+k)*nT] = -1
        bcdx[k, (nx+k)*nT+1] = 1
        fbcdx[k] = h1*inidx[k]

    # on l : l(T) = 0
    bcl = np.zeros((nx, (nx+nx+nu)*nT))
    fbcl = np.zeros((nx, 1))
    for k in range(nx):
        bcl[k, (k+1)*nT-1] = 1
        fbcl[k] = 0

    # on dl : dl(T) = -gamma*C.T*(Cx(T)-g(T))
    bcdl = np.zeros((nx, (nx+nx+nu)*nT))
    fbcdl = np.zeros((nx, 1))
    hN = tmesh[-1] - tmesh[-2]
    for k in range(nx):
        bcdl[k, (k+1)*nT-2] = -1
        bcdl[k, (k+1)*nT-1] = 1
        for i in range(C.shape[1]):
            bcdl[k, (nx+i+1)*nT-1] = gamma*hN*np.dot(C.T, C)[k, i]
        fbcdl[k] = gamma*hN*(C.T)[k]*g(tmesh[-1])

    # on du : du(0) = 0
    bcduz = np.zeros((nu, (nx+nx+nu)*nT))
    fbcduz = np.zeros((nu, 1))
    if udiril[0]:  # u(0) = 0
        for k in range(nu):
            bcduz[k, (nx+nx+k)*nT] = 1
            fbcduz[k] = 0
    else:  # or du(0) = 0
        for k in range(nu):
            bcduz[k, (nx+nx+k)*nT] = -1
            bcduz[k, (nx+nx+k)*nT+1] = 1
            fbcduz[k] = 0

    # on du : du(T) = 0
    bcdut = np.zeros((nu, (nx+nx+nu)*nT))
    fbcdut = np.zeros((nu, 1))
    if udiril[0]:  # u(0) = 0
        for k in range(nu):
            bcdut[k, (nx+nx+k+1)*nT-1] = 1
            fbcdut[k] = 0
    else:
        for k in range(nu):
            bcdut[k, (nx+nx+k+1)*nT-2] = -1
            bcdut[k, (nx+nx+k+1)*nT-1] = 1
            fbcdut[k] = 0

    bccoff = np.vstack([bcx, bcdx, bcl, bcdl, bcduz, bcdut])
    bcrhs = np.vstack([fbcx, fbcdx, fbcl, fbcdl, fbcduz, fbcdut])

    coeff = np.vstack([bigdiff-bigcoff, bccoff])
    rhs = np.vstack([rhsinner, bcrhs])

    sol = npla.solve(coeff, rhs)
    return sol


def ltv_holo_tpbvfindif(tmesh=None, mmat=None, bmat=None, inpufun=None,
                        getgmat=None, getamat=None, xini=None, vini=None,
                        xrhs=None, nr=None):
    ''' model structure
    Mx'' = Ax - G.Tp + Bu + rhs
       0 = Gx
    with A, G, rhs time-varying
    '''

    diffmat = disc_cent_diffs(tmesh)
    spdfm = sps.csc_matrix(diffmat)
    tdmasmat = sps.kron(spdfm, mmat)

    nx = xini.size
    ntp = len(tmesh)
    intmesh = tmesh[1:-1]
    ntpi = len(intmesh)
    tsiinv = 1./(tmesh[1] - tmesh[0])

    def _zspm(nl, nc):
        return sps.csc_matrix((nl, nc))

    adiag, gdiag, rhslist = [], [], []
    for curt in intmesh:
        adiag.append(sps.csc_matrix(getamat(curt)))
        gdiag.append(getgmat(curt))
        rhslist.append((bmat.dot(inpufun(curt)) + xrhs(curt)).reshape((nx, )))

    tdamat = sps.hstack([_zspm(ntpi*nx, nx),
                         sps.block_diag(adiag), _zspm(ntpi*nx, nx)])

    tdgmat = sps.block_diag(gdiag)
    tdrhs = np.hstack([rhslist]).reshape((ntpi*nx, 1))

    """ what the coeffmat will look like:

    dims        nx   ntp-1*nx   ntp-2*nr
               -------------------------
             |
    nx       |  inixm   0       0
    nx       |  inivm   0       0
    ntp-2*nx |  --coefmatx--    gmat.T
    ntp-2*nr |  0       gmat    0

    by now no constraint/multiplier at t=T
    """

    inixmat = sps.hstack([sps.eye(nx), _zspm(nx, (ntp-1)*nx+ntpi*nr)])
    inivmat = tsiinv*sps.hstack([-sps.eye(nx), sps.eye(nx),
                                 _zspm(nx, ntpi*(nx+nr))])
    coefmatx = sps.hstack([tdmasmat-tdamat, tdgmat.T])
    coefmatg = sps.hstack([_zspm(ntpi*nr, nx), tdgmat,
                           _zspm(ntpi*nr, nx+ntpi*nr)])

    coefmat = sps.vstack([inixmat, inivmat, coefmatx, coefmatg]).tocsc()
    rhsx = np.vstack([xini, vini, tdrhs, np.zeros((ntpi*nr, 1))])

    ucomat = coefmat[:ntp*nx, :][:, :ntp*nx]
    gmat = coefmat[ntp*nx:, :][:, :ntp*nx]
    tgmat = coefmat[:ntp*nx, :][:, ntp*nx:]
    ucomati = np.linalg.inv(ucomat.todense())
    scc = gmat*(tgmat.T*ucomati.T).T
    print np.linalg.cond(scc)
    print np.linalg.cond(ucomat.todense())
    print np.linalg.cond(coefmat.todense())

    import matplotlib.pyplot as plt
    plt.figure(11)
    plt.spy(ucomat)
    plt.figure(12)
    plt.spy(gmat)
    plt.figure(13)
    plt.spy(tgmat)
    # plt.figure(1)
    # plt.spy(inixmat)
    # plt.figure(2)
    # plt.spy(inivmat)
    # plt.figure(3)
    # plt.spy(coefmatx)
    # plt.figure(4)
    # plt.spy(coefmatg)
    # plt.figure(5)
    # plt.spy(coefmat)
    # plt.figure(6)
    # plt.spy(tdmasmat)
    # plt.figure(7)
    # plt.spy(tdamat)
    plt.show(block=False)
    # raise Warning('TODO: debug')

    tdsol = spsla.spsolve(coefmat, rhsx)

    return tdsol


if __name__ == '__main__':
    # tmesh = np.array([0., 0.1, 0.3, 0.4, 0.8, 0.9, 1.]).reshape((7, 1))
    mesh = np.linspace(0, 1, 7).reshape((7, 1))
    disc_cent_diffs(mesh)
