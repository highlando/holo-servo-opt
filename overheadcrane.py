import numpy as np
import scipy.optimize as sco

import probdefs as pbd
import matplotlib.pyplot as plt

import ohc_utils as ocu
import ohc_plot_utils as plu

'''
the general model structure is

Mx'' = Ax - G.T(x)l + Bu + f
g(x) = 0
'''


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
    plist = [0]  # list for the multiplier
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
        plist.append(xvpqnew[-2:-1, ])  # TODO: softcode plz

    return ylist, ulist, plist


def bwsweep(tmesh=None, amatfun=None, rhsfun=None, gmatfun=None, mmat=None,
            terml=None, termn=None, outputmat=None):
    curG = gmatfun(tmesh[-1])
    if not np.allclose(curG.dot(terml), 0):
        print 'need to project the inivals'
    if not np.allclose(curG.dot(termn), 0):
        print 'need to project the inivals'
    curl, curn = terml, termn
    ulist = [outputmat.dot(curl)]
    (nr, nx) = curG.shape
    for k, curt in enumerate(reversed(tmesh[:-1])):
        cts = tmesh[-k] - curt
        preA = amatfun(curt)
        preG = gmatfun(curt)
        cfm1 = np.hstack((1./cts*mmat, -preA, preG.T))
        cfm2 = np.hstack((-mmat, 1./cts*mmat, np.zeros((nx, nr))))
        cfm3 = np.hstack((np.zeros((nr, nx)), -preG, np.zeros((nr, nr))))
        cfm = np.vstack((cfm1, cfm2, cfm3))
        prerhs = np.vstack((1./cts*mmat.dot(curn)-rhsfun(curt),
                            1./cts*mmat.dot(curl),
                            0))
        prenlm = np.linalg.solve(cfm, prerhs)
        curl, curn = prenlm[nx:2*nx, :], prenlm[:nx, :]
        ulist.append(outputmat.dot(curl))
    ulist.reverse()
    return ulist


if __name__ == '__main__':
    tE, Nts = 3., 30
    tmesh = np.linspace(0, tE, Nts).tolist()
    # defining the target trajectory and the exact solution
    inix = np.array([[0, 40, 0, 4]]).T
    iniv = np.array([[0., 0., 0., 0.]]).T
    nx, ny, nu = inix.size, 2, 2
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
    # def of the optimization problem
    qmat = np.eye(ny)
    beta = 1e-7
    rmatinv = 1./beta*np.eye(nu)
    gamma = 1e-3
    smat = gamma*np.eye(ny)
    # the data of the problem
    ovhdcrn = ocu.overheadmodel(**modpardict)
    mmat, cmat, bmat = ovhdcrn['mmat'], ovhdcrn['cmat'], ovhdcrn['bmat']
    exatinp = ocu.get_exatinp(scalarg=scalarg, gm0=gm0, gmf=gmf, **modpardict)

    def ystar(t):
        return ocu.trgttrj(t, scalarg=scalarg,
                           gm0=gm0, gmf=gmf, retdts_all=False)
    ovhdcrn.update(dict(cmat=None))
    xlist, ulist, plist = \
        int_impeul_ggl(inix=inix, iniv=iniv, inpfun=exatinp,
                       tmesh=tmesh, **ovhdcrn)
    plu.plotxlist(xlist, tmesh=tmesh)

    xld, pld = dict(zip(tmesh, xlist)), dict(zip(tmesh, plist))

    def get_getbwamat(pld=None, amat=None, r=None):
        def getbwamat(t):
            curp = pld[t]
            if amat is not None:
                raise NotImplementedError('...')
            else:
                return -2*curp*np.array([[1, 0, -1, 0],
                                         [0, -r**2, 0, 0],
                                         [-1, 0, 1, 0],
                                         [0, 0, 0, 1]])
        return getbwamat

    def get_getgmat(xld=None, holojaco=None):
        def getgmat(t):
            curx = xld[t].reshape((nx, 1))
            return holojaco(curx)
        return getgmat

    def get_getdualrhs(cmat=None, qmat=None, trgttrj=None, xld=None):
        def getdualrhs(t):
            curx = xld[t].reshape((nx, 1))
            curey = trgttrj(t)
            return -np.dot(cmat.T, np.dot(qmat, np.dot(cmat, curx)-curey))
        return getdualrhs

    getbwamat = get_getbwamat(pld=pld, r=r)
    getdualrhs = get_getdualrhs(cmat=cmat, qmat=qmat,
                                trgttrj=ystar, xld=xld)
    getgmat = get_getgmat(xld=xld, holojaco=ovhdcrn['holojaco'])

    tbmatt = rmatinv.dot(bmat.T)
    terml = np.zeros((nx, 1))
    gettermn = get_getdualrhs(cmat=cmat, qmat=smat, trgttrj=ystar, xld=xld)
    termn = np.linalg.solve(mmat, gettermn(tmesh[-1]))
    ulist = bwsweep(tmesh=tmesh, amatfun=getbwamat, rhsfun=getdualrhs,
                    gmatfun=getgmat, mmat=mmat,
                    terml=terml, termn=termn, outputmat=tbmatt)
    # plu.plttrjtrj(trgttrj)

    plt.show(block=False)
