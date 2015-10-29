import numpy as np
import scipy.optimize as sco

import probdefs as pbd
import matplotlib.pyplot as plt


def overheadmodel(J=None, m=None, mt=None, r=None, gravity=9.81):
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
    ovhdcrn = overheadmodel(J=.1, m=100., mt=10., r=.1)
    """

    mmat = np.diag([mt, J, m, m])

    # state: x = [s, beta, xd, zd].T

    amat = np.zeros((4, 4))
    bmat = np.array([[1., 0], [0, 1.], [0, 0], [0, 0]])
    cmat = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])

    rhs = np.array([[0, 0, 0, m*gravity]]).T

    def holoc(x=None):
        return (x[2] - x[0])**2 + x[3]**2 - (r*x[1])**2

    def holojaco(x):
        return 2*np.array([[-(x[2]-x[0]), -r**2*x[1], x[2]-x[0], x[3]]])

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
            - rhs.flatten() - bmat.dot(uvec).flatten()
        resthr = holoc(xcur)
        resfou = np.dot(cgholo, vcur)

        res = np.r_[resone, restwo, resthr, resfou]

        return res

    if cmat is not None:
        ylist = [np.dot(cmat, inix)]
    else:
        ylist = [inix]
    xold, vold = inix.flatten(), iniv.flatten()
    xvpqold = np.vstack([inix, iniv, np.array([[0], [0]])])
    ulist, glist = [inpfun(tmesh[0])], [holoc(xold)]
    for k, tk in enumerate(tmesh[1:]):
        uvec = inpfun(tk)
        # print 'u({0})'.format(tk), uvec
        dt = tk - tmesh[k]

        def _optires(xvpq):
            return _imp_ggl_res(xvpq, xold=xold, vold=vold, uvec=uvec, dt=dt)
        xvpqnew = sco.fsolve(_optires, xvpqold)
        xold, vold = xvpqnew[:nx, ], xvpqnew[nx:2*nx, ]
        if cmat is not None:
            ylist.append(np.dot(cmat, xold))
        else:
            ylist.append(xold)
        ulist.append(inpfun(tk))
        glist.append(holoc(xold))

    return ylist, ulist, glist

if __name__ == '__main__':
    tE, Nts = 3., 1200
    # defining the target trajectory and the exact solution
    gm0 = np.array([[0., 4.]]).T
    gmf = np.array([[5., 1.]]).T
    gmf = np.array([[0., 2.]]).T

    scalarg = pbd.get_trajec('pwp', tE=tE, g0=0., gf=1.,
                             trnsarea=tE, polydeg=9, retdts_all=True)

    def trgttrj(t, retdts_all=True):
        sclrgt = scalarg(t)  # , retdts_all=retdts_all)
        if retdts_all:
            return (gm0 + sclrgt[0]*(gmf - gm0),
                    + sclrgt[1]*(gmf - gm0),
                    + sclrgt[2]*(gmf - gm0),
                    + sclrgt[3]*(gmf - gm0),
                    + sclrgt[4]*(gmf - gm0))
        else:
            return gm0 - sclrgt[0]*(gmf - gm0)

    tmesh = np.linspace(0, tE, Nts).tolist()
    inix = np.array([[0, 40, 0, 4]]).T
    iniv = np.array([[0., 0., 0., 0.]]).T
    modpardict = dict(J=.1, m=100., mt=10., r=.1, gravity=9.81)

    J, m, mt = modpardict['J'], modpardict['m'], modpardict['mt']
    r, gravity = modpardict['r'], modpardict['gravity']

    def excatinp(t):
        def _auxvals(curt):
            ctg = trgttrj(curt, retdts_all=True)
            g1, g2 = ctg[0][0], ctg[0][1]
            d1g1, d2g1, d3g1, d4g1 = ctg[1][0], ctg[2][0], ctg[3][0], ctg[4][0]
            d1g2, d2g2, d3g2, d4g2 = ctg[1][1], ctg[2][1], ctg[3][1], ctg[4][1]

            # define aux variables as in rob Alt, Betsch, Yang '14 w/ exact sol
            gmd2g2 = gravity - d2g2
            lmbd = .5*m/g2*gmd2g2
            s = g1 + d2g1*g2/gmd2g2
            bt = + g2/(r*gmd2g2)*np.sqrt(d2g1**2+gmd2g2**2)
            tt = d1g1 + d2g1*g2*d3g2/gmd2g2**2 + (d3g1*g2+d2g1*d1g2)/gmd2g2
            alpha = (d2g1*tt - d2g1*d1g1 + d1g2*gmd2g2) /\
                (r*np.sqrt(d2g1**2+gmd2g2**2))
            th = d2g1 + 2*d2g1*g2*d3g2**2/gmd2g2**3 +\
                (2*d3g1*g2*d3g2+2*d2g1*d1g2*d3g2+d2g1*g2*d4g2)/gmd2g2**2 +\
                (d4g1*g2+2*d3g1*d1g2+d2g1*d2g2)/gmd2g2
            halp = ((s-g1)*th + (g1-s)*d2g1 + g2*d2g2 + (tt-d1g1)*tt -
                    r**2*alpha**2 + (d1g1-tt)*d1g1 + d1g2**2)/(r**2*bt)
            return th, lmbd, halp, bt, s, g1
        th, lmbd, halp, bt, s, g1 = _auxvals(t)
        uF = mt*th + 2*(s-g1)*lmbd
        uM = J*halp - 2*r**2*bt*lmbd

        return np.array([uF, uM])

    def testinp(t):
        mpd = modpardict
        return np.array([0, -mpd['m']*mpd['gravity']*mpd['r']])

    uflist, umlist = [], []
    for tk in tmesh:
        uflist.append(excatinp(tk)[0][0])
        umlist.append(excatinp(tk)[1][0])

    plt.figure(1)
    plt.plot(tmesh, uflist)
    plt.figure(2)
    plt.plot(tmesh, umlist)

    ovhdcrn = overheadmodel(**modpardict)
    # ovhdcrn.update(dict(cmat=None))
    xlist, ulist, reslist = \
        int_impeul_ggl(inix=inix, iniv=iniv, inpfun=excatinp,
                       tmesh=tmesh, **ovhdcrn)

    def plotxlist(xlist, tmesh=None):
        posarray = np.r_[xlist]
        plt.figure(123)
        plt.plot(posarray[:, 0], posarray[:, 1])
        if tmesh is not None:
            plt.figure(124)
            plt.plot(tmesh, posarray[:, 0])
            plt.figure(125)
            plt.plot(tmesh, posarray[:, 1])

    plotxlist(xlist, tmesh=tmesh)

    def plttrjtrj(gfun):
        ndts = 4
        xdsl, zdsl = [], []
        gt = gfun(tmesh[0])
        for k in range(ndts):
            xdsl.append([gt[k][0][0]])
            zdsl.append([gt[k][1][0]])
        for tk in tmesh[1:]:
            gt = gfun(tk)
            for k in range(ndts):
                xdsl[k].append(gt[k][0][0])
                zdsl[k].append(gt[k][1][0])
        for k in range(ndts):
            plt.figure(211)
            plt.plot(tmesh, xdsl[k])
        plt.figure(212)
        for k in range(ndts):
            plt.plot(tmesh, zdsl[k])
    # plttrjtrj(trgttrj)

    plt.show(block=False)
