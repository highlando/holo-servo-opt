import numpy as np
import scipy.optimize as sco

import probdefs as pbd
import ohc_utils as ocu
import bswptest as bst

# import matplotlib.pyplot as plt
# import ohc_plot_utils as plu


'''
the general model structure is

Mx'' = Ax - G.T(x)p + Bu + f
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
        # cfm1 = np.hstack((1./cts*mmat, -preA, preG.T, np.zeros((nx, nr))))
        # cfm2 = np.hstack((-mmat, 1./cts*mmat, np.zeros((nx, nr)), preG.T))
        # cfm3 = np.hstack((np.zeros((nr, nx)), -preG, np.zeros((nr, 2*nr))))
        # cfm4 = np.hstack((-preG, np.zeros((nr, nx)), np.zeros((nr, 2*nr))))
        # cfm = np.vstack((cfm1, cfm2, cfm3, cfm4))
        # prerhs = np.vstack((1./cts*mmat.dot(curn)+rhsfun(curt),
        #                     1./cts*mmat.dot(curl),
        #                     0,
        #                     0))

        upd = bst.scndordbwstep(amat=preA, mmat=mmat, gmat=preG,
                                lini=terml, dlini=-curn,
                                rhs=rhsfun(curt), ts=cts)
        curl = upd[:nx, :]
        curn = upd[nx: 2*nx, :]

        # prenlm = np.linalg.solve(cfm, prerhs)
        # curl, curn = prenlm[nx:2*nx, :], prenlm[:nx, :]

        # raise Warning('TODO: debug')

        ulist.append(outputmat.dot(curl))
    ulist.reverse()
    return ulist


if __name__ == '__main__':
    tE, Nts = 3., 301
    tmesh = np.linspace(0, tE, Nts).tolist()
    # defining the target trajectory and the exact solution
    inix = np.array([[0, 40, 0, 4]]).T
    iniv = np.array([[0., 0., 0., 0.]]).T
    nx, ny, nu = inix.size, 2, 2
    # initial and final point
    gm0 = np.array([[0., 4.]]).T
    gmf = np.array([[0., 5.]]).T
    # gmf = np.array([[0., 4.1]]).T
    # gmf = np.array([[0., 5.]]).T
    # scalar morphing function
    scalarg = pbd.get_trajec('pwp', tE=tE, g0=0., gf=1.,
                             trnsarea=tE, polydeg=9, retdts_all=True)
    # def of the problem
    modpardict = dict(J=.1, m=100., mt=10., r=.1, gravity=9.81)
    J, m, mt = modpardict['J'], modpardict['m'], modpardict['mt']
    r, gravity = modpardict['r'], modpardict['gravity']
    # def of the optimization problem
    qmat = np.eye(ny)
    beta = 1e-4
    rmatinv = 1./beta*np.eye(nu)
    gamma = 1e-3
    smat = gamma*np.eye(ny)
    # the data of the problem
    ovhdcrn = ocu.overheadmodel(**modpardict)
    mmat, cmat, bmat = ovhdcrn['mmat'], ovhdcrn['cmat'], ovhdcrn['bmat']
    minv = np.linalg.inv(mmat)
    exatinp = ocu.get_exatinp(scalarg=scalarg, gm0=gm0, gmf=gmf, **modpardict)

    def ystar(t):
        return ocu.trgttrj(t, scalarg=scalarg,
                           gm0=gm0, gmf=gmf, retdts_all=False)
    ovhdcrn.update(dict(cmat=None))

    def zeroinp(t):
        return np.zeros((2, 1))

    def keepitconst(t):  # for constant position
        return np.array([[0, -m*gravity*r]]).T
    xlist, ulist, plist = \
        int_impeul_ggl(inix=inix, iniv=iniv,
                       # inpfun=zeroinp,
                       # inpfun=testinp,
                       inpfun=keepitconst,
                       tmesh=tmesh, **ovhdcrn)
    # plu.plotxlist(xlist, tmesh=tmesh)

    picardsteps = 3
    for npc in range(picardsteps):
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
                drhs = -np.dot(cmat.T, np.dot(qmat, np.dot(cmat, curx)-curey))
                return drhs
            return getdualrhs

        getbwamat = get_getbwamat(pld=pld, r=r)
        getdualrhs = get_getdualrhs(cmat=cmat, qmat=qmat,
                                    trgttrj=ystar, xld=xld)
        getgmat = get_getgmat(xld=xld, holojaco=ovhdcrn['holojaco'])

        tbmatt = rmatinv.dot(bmat.T)
        terml = np.zeros((nx, 1))
        gettermld = get_getdualrhs(cmat=cmat, qmat=smat,
                                   trgttrj=ystar, xld=xld)
        termld = -np.linalg.solve(mmat, gettermld(tmesh[-1]))

        # tend = tmesh[-1]
        # print xld[tend]
        # print ystar(tend)
        # ## make the terminal value consistent
        curG = getgmat(tmesh[-1])
        minvGt = np.dot(minv, curG.T)
        csc = np.dot(curG, minvGt)
        prjtermld = termld - \
            np.dot(minvGt, np.linalg.solve(csc, np.dot(curG, termld)))
        # raise Warning('TODO: debug')

        ulstt = bwsweep(tmesh=tmesh, amatfun=getbwamat, rhsfun=getdualrhs,
                        gmatfun=getgmat, mmat=mmat,
                        terml=terml, termld=prjtermld, outputmat=tbmatt)
        uld = dict(zip(tmesh, ulstt))

        def curinp(t):
            return uld[t].reshape((2, 1)) + keepitconst(t)

        xlist, curulist, plist = \
            int_impeul_ggl(inix=inix, iniv=iniv,
                           # inpfun=testinp,
                           inpfun=curinp,
                           tmesh=tmesh, **ovhdcrn)

    # plu.plttrjtrj(trgttrj)
    vbeta, vzd = inix[1], inix[3]
    vp = pld[tmesh[-1]]
    mmat, amat, gmat = bst.get_magmats(J=J, m=m, beta=vbeta, zd=vzd,
                                       p=vp, r=r)
    l2l4inds = np.array([False, True, False, True])
    lini = terml[l2l4inds]
    dlini = prjtermld[l2l4inds]
    cts = tmesh[-1] - tmesh[-2]
    rhs = getdualrhs(tmesh[-1])[l2l4inds]
    upd = bst.scndordbwstep(amat=amat, mmat=mmat, gmat=gmat,
                            lini=lini, dlini=dlini,
                            rhs=rhs, ts=cts)

    # plt.show(block=False)

    # curl, curn = terml, -termld
    # (nr, nx) = curG.shape
    # acz = J*vzd/(r**2*vbeta) + m*r**2*beta
    # aco = 2*vp*(vzd/(r*vbeta) - r**2*vbeta/vzd)
    # zcur, ltcur = -prjtermld[1], terml[1]
    # for k, curt in enumerate(reversed(tmesh[:-1])):
    #     cts = tmesh[-k] - curt
    #     coefm = np.array([[-1., 1/cts], [1./cts, -acz/aco]])
    #     prhs = np.vstack([[ltcur/cts], [zcur/cts + getdualrhs(curt)[1]]])
    #     zlp = np.linalg.solve(coefm, prhs)
    #     zcur, ltcur = zlp[0], zlp[1]
    #     print 1./beta*ltcur
