import numpy as np
# import scipy as sp


def bwsweep(tmesh=None, amatfun=None, rhsfun=None, gmatfun=None,
            mmat=None, terml=None, termld=None, outputmat=None):
    curG = gmatfun(tmesh[-1])
    if not np.allclose(curG.dot(terml), 0):
        print 'need to project the inivals'
    if not np.allclose(curG.dot(termld), 0):
        print 'need to project the inivals'
    curl, curn = terml, -termld
    ulist = [outputmat.dot(curl)]
    (nr, nx) = curG.shape
    for k, curt in enumerate(reversed(tmesh[:-1])):
        cts = tmesh[-k-1] - curt
        preA = amatfun(curt)
        preG = gmatfun(curt)

        upd = scndordbwstep(amat=preA, mmat=mmat, gmat=preG,
                            lini=terml, dlini=-curn,
                            rhs=rhsfun(curt), ts=cts)

        curl = upd[:nx, :]
        curn = upd[nx: 2*nx, :]

        ulist.append(outputmat.dot(curl))
    ulist.reverse()
    return ulist


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
