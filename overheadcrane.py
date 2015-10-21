import numpy as np
import scipy.optimize as sco

import probdefs as pbd


def overheadmodel(J=None, m=None, mt=None, r=1.):
    """ provide the model of the overhead crane

    described by the state `x` with

     * `x[0]:= s` - the x position of the cart
     * `x[1]:= beta` - modelling how pulling of the rope
     * `x[2]:= xd` - the x-position of the load
     * `x[3]:= zd` - the y-position of the load

    Returns
    ---
    ovhdcrn : dict
        with the following keys
         * `mmat`: the mass matrix
         * `amat`: the stiffness matrix
         * `bmat`: the input matrix
         * `holoc`: callable `g(x)` ret. the value of the constraint at `x`
         * `holojaco`: callable `G(x)` returning the Jacobi matrix at `x`

    Examples
    ---
    ovhdcrn = overheadmodel(J=1., m=1., mt=1., r=1.)
    """

    mmat = np.diag([mt, J/r**2, m, m])

    # state: x = [s, beta, xd, zd].T

    amat = np.zeros((4, 4))
    bmat = np.array([[1, 0], [0, 1./r], [0, 0], [0, 0]])
    cmat = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])

    rhs = np.array([[0, 0, 0, -m*9.81]]).T

    def holoc(x=None):
        return (x[2] - x[0])**2 + x[3]**2 - x[1]**2

    def holojaco(x):
        return 2*np.array([[-(x[2]-x[0]), -x[1], x[2]-x[0], x[3]]])

    ovhdcrn = dict(mmat=mmat, amat=amat, bmat=bmat, cmat=cmat,
                   rhs=rhs, holoc=holoc, holojaco=holojaco)
    return ovhdcrn


def int_impeul_ggl(mmat=None, amat=None, rhs=None, holoc=None, holojaco=None,
                   bmat=None, inpfun=None, cmat=None,
                   inix=None, iniv=None, tmesh=None):
    """ integrate mbs with holonomic constraints using imp euler

    and the Gear-Gupta-Leimkuhler Index-2 formulation"""

    nx = mmat.shape[0]

    def _imp_ggl_res(xvpq, xold=None, vold=None, uvec=None, dt=None):
        xcur = xvpq[:nx]
        vcur = xvpq[nx:2*nx]
        pcur = xvpq[-2:-1]
        qcur = xvpq[-1:]
        cgholo = holojaco(xcur)
        resone = 1./dt*np.dot(mmat, xcur-xold) - np.dot(mmat, vcur) \
            + np.dot(cgholo.T, qcur)
        restwo = 1./dt*np.dot(mmat, vcur-vold) + np.dot(cgholo.T, pcur) \
            - rhs.flatten()
        resthr = holoc(xcur)
        resfou = np.dot(cgholo, vcur)

        res = np.r_[resone, restwo, resthr, resfou]

        return res

    if cmat is not None:
        ylist = [np.dot(cmat, inix)]
    else:
        ylist = [inix]
    xold, vold = inix.flatten(), iniv.flatten()
    print 'g(xcur) = ', holoc(xold)
    xvpqold = np.vstack([inix, iniv, np.array([[0], [0]])])
    for k, tk in enumerate(tmesh[1:]):
        uvec = inpfun(tk)
        dt = tk - tmesh[k]

        def _optires(xvpq):
            return _imp_ggl_res(xvpq, xold=xold, vold=vold, uvec=uvec, dt=dt)
        xvpqnew = sco.fsolve(_optires, xvpqold)
        xold, vold = xvpqnew[:nx, ], xvpqnew[nx:2*nx, ]
        print 'g(xcur) = ', holoc(xold)
        if cmat is not None:
            ylist.append(np.dot(cmat, xold))
        else:
            ylist.append(xold)

    return ylist

if __name__ == '__main__':
    tE, Nts = 3., 3000
    # defining the target trajectory and the exact solution
    gm0 = np.array([[0., 4.]]).T
    gmf = np.array([[1., 5.]]).T

    scalarg = pbd.get_trajec('pwp', tE=tE, g0=1., gf=1.,
                             trnsarea=tE, polydeg=9, retderivs=True)

    def trgttrj(t, retderivs=False):
        sclrgt = scalarg(t)
        if retderivs:
            return (gm0 - sclrgt[0]*(gmf - gm0),
                    gm0 - sclrgt[1]*(gmf - gm0),
                    gm0 - sclrgt[2]*(gmf - gm0),
                    gm0 - sclrgt[3]*(gmf - gm0))
        else:
            return gm0 - sclrgt[0]*(gmf - gm0)

    tmesh = np.linspace(0, 0.1, 100).tolist()
    inix = np.array([[0, 5, 3, -4]]).T
    iniv = np.array([[0., 0., 0., 0.]]).T

    def inpu(t):
        return 0*np.array([[1, 0]]).T

    ovhdcrn = overheadmodel(J=1., m=1., mt=1., r=1.)
    ovhdcrn.update(dict(cmat=None))
    xlist = int_impeul_ggl(inix=inix, iniv=iniv, inpfun=inpu, tmesh=tmesh,
                           **ovhdcrn)
