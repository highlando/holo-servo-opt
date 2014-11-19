import numpy as np


def disc_cent_diffs(tmesh):
    N = tmesh.size
    # define the discrete difference matrix
    hvec = tmesh[1:] - tmesh[:-1]
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
               tmesh=None, Q=None, bone=None, bzero=None):
    """ finite differences for the second order optimality system

    0  I  0                     -  0     A  B               = f
    I  0  0   * [l'', x'', u''] -  A.T  -Q  0   * [l, x, u] = g
    0  0 b1*I                   -  -B.T  0 b0*I             = 0
    """

    diffmat = disc_cent_diffs(tmesh)
    intmesh = tmesh[1:-1]
    nint = intmesh.size
    nx = A.shape[0]
    nu = B.shape[0]
    gvec = disc_func(g, intmesh)
    ftl = []
    for f in flist:
        ftl.append(disc_func(f, intmesh))
    fvec = np.vstack(ftl)
    rhsinner = np.vstack([fvec, np.kron(C.T, gvec), np.zeros((nint, 1))])

    xxz = np.zeros((nx*nint, nx*(nint+2)))
    xuz = np.zeros((nx*nint, nu*(nint+2)))
    uxz = np.zeros((nu*nint, nx*(nint+2)))

    xleye = np.eye((nx, nx))
    ueye = np.eye((nu, nu))
    diffxl = np.kronecker(xleye, diffmat)
    diffu = np.kronecker(ueye, diffmat)
    xdiff = np.hstack([xxz, diffxl, xuz])
    ldiff = np.hstack([diffxl, xxz, xuz])
    udiff = np.hstack([uxz, uxz, diffu])
    bigdiff = np.vstack([xdiff, ldiff, udiff])

    eyetint = np.eye(tmesh.size)[1:-1, :]
    xxcoff = np.kronecker(A, eyetint)
    xucoff = np.kronecker(B, eyetint)
    llcoff = np.kronecker(A.T, eyetint)
    lxcoff = np.kronecker(-Q, eyetint)
    ulcoff = np.kronecker(-B.T, eyetint)

    uucoff = bzero*np.kronecker(ueye, eyetint)

    bigcoff = np.vstack([np.hstack([xxz, xxcoff, xucoff]),
                         np.hstack([llcoff, lxcoff, xuz]),
                         np.hstack([ulcoff, uxz, uucoff])])

    return

if __name__ == '__main__':
    # tmesh = np.array([0., 0.1, 0.3, 0.4, 0.8, 0.9, 1.]).reshape((7, 1))
    tmesh = np.linspace(0, 1, 7).reshape((7, 1))
    disc_cent_diffs(tmesh)
