import numpy as np
import scipy as sp


def get_magmats(J=None, p=None, r=None, m=None, beta=None, zd=None):

    mmat = np.diag([J, m])
    prsqr, mtp = 2*p*r**2, -2*p
    amat = np.diag([prsqr[0], mtp[0]])
    # amat = sp.linalg.block_diag([2*p*r**2, -2*p])
    gmat = 2*np.array([[-r**2*beta, zd]]).reshape((1, 2))
    return mmat, amat, gmat


def scndordbwstep(amat=None, mmat=None, gmat=None, lini=None, dlini=None,
                  rhs=None, ts=None):

    vini = -dlini
    nr, nx = gmat.shape
    como = np.hstack([1./ts*mmat, mmat, 0*gmat.T])
    comt = np.hstack([-amat, 1./ts*mmat, gmat.T])
    comz = np.hstack([gmat, 0*gmat, np.zeros((nr, nr))])

    comat = np.vstack([como, comt, comz])

    crhs = np.vstack([1./ts*mmat.dot(lini),
                      1./ts*mmat.dot(vini) + rhs,
                      np.zeros((nr, 1))])

    return np.linalg.solve(comat, crhs)
