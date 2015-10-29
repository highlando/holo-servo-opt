import numpy as np
import scipy.optimize as sco

import probdefs as pbd
import matplotlib.pyplot as plt

import ohc_utils as ocu


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
    tmesh = np.linspace(0, tE, Nts).tolist()
    # defining the target trajectory and the exact solution
    inix = np.array([[0, 40, 0, 4]]).T
    iniv = np.array([[0., 0., 0., 0.]]).T
    # initial and final point
    gm0 = np.array([[0., 4.]]).T
    gmf = np.array([[5., 1.]]).T
    # scalar morphing function
    scalarg = pbd.get_trajec('pwp', tE=tE, g0=0., gf=1.,
                             trnsarea=tE, polydeg=9, retdts_all=True)
    # def of the problem
    modpardict = dict(J=.1, m=100., mt=10., r=.1, gravity=9.81)
    J, m, mt = modpardict['J'], modpardict['m'], modpardict['mt']
    r, gravity = modpardict['r'], modpardict['gravity']
    # the data of the problem
    ovhdcrn = ocu.overheadmodel(**modpardict)
    exatinp = ocu.get_exatinp(scalarg=scalarg, gm0=gm0, gmf=gmf, **modpardict)

    uflist, umlist = [], []
    for tk in tmesh:
        uflist.append(exatinp(tk)[0][0])
        umlist.append(exatinp(tk)[1][0])

    plt.figure(1)
    plt.plot(tmesh, uflist)
    plt.figure(2)
    plt.plot(tmesh, umlist)

    # ovhdcrn.update(dict(cmat=None))
    xlist, ulist, reslist = \
        int_impeul_ggl(inix=inix, iniv=iniv, inpfun=exatinp,
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
