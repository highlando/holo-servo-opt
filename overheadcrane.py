import numpy as np
import scipy.optimize as sco

import probdefs as pbd
import ohc_utils as ocu
import bswptest as bst

# import seco_order_opti as soo
import first_order_opti as foo
# import matplotlib.pyplot as plt
# import ohc_plot_utils as plu


'''
the general model structure is

Mx'' = Ax - G.T(x)p + Bu + f
g(x) = 0
'''


def int_impeul_ggl(mmat=None, amat=None, rhs=None, holoc=None, holojaco=None,
                   bmat=None, inpfun=None, cmat=None,
                   inix=None, iniv=None, tmesh=None, retvlist=False):
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
        ylist = [np.dot(cmat, inix).flatten()]
    else:
        ylist = [inix.flatten()]
        vlist = [iniv.flatten()]
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
            vlist.append(vold)
        ulist.append(inpfun(tk))
        glist.append(holoc(xold))
        plist.append(xvpqnew[-2:-1, ])  # TODO: softcode plz

    if retvlist:
        return ylist, ulist, plist, vlist

    else:
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

        upd = bst.scndordbwstep(amat=preA, mmat=mmat, gmat=preG,
                                lini=terml, dlini=-curn,
                                rhs=rhsfun(curt), ts=cts)

        curl = upd[:nx, :]
        curn = upd[nx: 2*nx, :]

        ulist.append(outputmat.dot(curl))
    ulist.reverse()
    return ulist


def get_pdxdxg(pld=None, amat=None, r=None):
    def pdxdxg(t):
        curp = pld[t]
        if amat is not None:
            raise NotImplementedError('...')
        else:
            return -2*curp*np.array([[1, 0, -1, 0],
                                     [0, -r**2, 0, 0],
                                     [-1, 0, 1, 0],
                                     [0, 0, 0, 1]])
    return pdxdxg


def get_getgmat(xld=None, holojaco=None):
    def getgmat(t):
        curx = xld[t].reshape((nx, 1))
        return holojaco(curx)
    return getgmat


def get_getdgmat(xld=None, vld=None, holohess=None):
    dxdxtg = 2*np.array([[1, 0, -1, 0],
                         [0, -r**2, 0, 0],
                         [-1, 0, 1, 0],
                         [0, 0, 0, 1]])

    def getdgmat(t):
        curv = vld[t].reshape((nx, 1))
        return np.dot(dxdxtg, curv).T
    return getdgmat


def get_getdualrhs(cmat=None, qmat=None, trgttrj=None, xld=None):
    def getdualrhs(t):
        curx = xld[t].reshape((nx, 1))
        curey = trgttrj(t)
        drhs = -np.dot(cmat.T, np.dot(qmat, np.dot(cmat, curx)-curey))
        return drhs
    return getdualrhs


def get_xresidual(xld=None, pld=None, mddxld=None, nx=None, NP=None,
                  holojaco=None, sysrhs=None, minusres=False):
    ''' returns a function that computes  Mx'' + G(x).Tp - f  at time t'''

    def xres(t):
        curx = xld[t].reshape((nx, 1))
        curp = pld[t].reshape((NP, 1))
        xres = mddxld[t].reshape((nx, 1)) + np.dot(holojaco(curx).T, curp) \
            - sysrhs
        if minusres:
            return -xres
        else:
            return xres

    return xres


def getmddxld(vld=None, mmat=None, tmesh=None):
    mddxld = {}
    for k, curt in enumerate(tmesh[1:]):
        pret = tmesh[k]
        ctsi = 1./(curt - pret)
        mddxld.update({curt: ctsi*np.dot(mmat, vld[curt] - vld[pret])})

    return mddxld


def get_grhs(xld=None, holoc=None):
    def grhs(t):
        return -holoc(xld[t])
    return grhs


def get_dgrhs(xld=None, vld=None, holojaco=None):
    def dgrhs(t):
        return -np.dot(holojaco(xld[t]), vld[t])
    return dgrhs


if __name__ == '__main__':
    tE, Nts = 3., 301
    tmesh = np.linspace(0, tE, Nts).tolist()
    # defining the target trajectory and the exact solution
    inix = np.array([[0, 40, 0, 4]]).T
    iniv = np.array([[0., 0., 0., 0.]]).T
    nx, ny, nu = inix.size, 2, 2
    NP = 1
    # initial and final point
    gm0 = np.array([[0., 4.]]).T
    # gmf = np.array([[1., 5.]]).T
    # gmf = np.array([[0., 4.1]]).T
    # gmf = np.array([[0., 5.]]).T
    gmf = np.array([[2., 5.]]).T

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

    xlist, ulist, plist, vlist = \
        int_impeul_ggl(inix=inix, iniv=iniv,
                       # inpfun=zeroinp,
                       inpfun=keepitconst,
                       # inpfun=exatinp,
                       tmesh=tmesh, retvlist=True, **ovhdcrn)
    # xldz, pldz = dict(zip(tmesh, xlist)), dict(zip(tmesh, plist))
    # vldz = dict(zip(tmesh, vlist))
    xold = np.hstack(xlist).reshape((Nts*nx, 1))

    linsteps = 5
    for npc in range(linsteps):
        xld, pld = dict(zip(tmesh, xlist)), dict(zip(tmesh, plist))
        vld = dict(zip(tmesh, vlist))
        mddxld = getmddxld(vld=vld, mmat=mmat, tmesh=tmesh)

        getgmat = get_getgmat(xld=xld, holojaco=ovhdcrn['holojaco'])
        getdgmat = get_getdgmat(vld=vld)
        getpdxdxg = get_pdxdxg(pld=pld, r=r)

        xrhs = get_xresidual(xld=xld, pld=pld, sysrhs=ovhdcrn['rhs'],
                             holojaco=ovhdcrn['holojaco'], mddxld=mddxld,
                             minusres=True, nx=nx, NP=NP)
        grhs = get_grhs(xld=xld, holoc=ovhdcrn['holoc'])
        dgrhs = get_dgrhs(xld=xld, vld=vld, holojaco=ovhdcrn['holojaco'])

        nr = 1
        xvqplmu = foo.\
            ltv_holo_tpbvfindif(tmesh=tmesh, mmat=mmat, bmat=bmat,
                                inpufun=exatinp, getgmat=getgmat,
                                getdgmat=getdgmat,
                                getamat=getpdxdxg, nr=nr,
                                grhs=grhs, dgrhs=dgrhs,
                                dxini=0*inix, dvini=0*iniv, xrhs=xrhs)

        ntp = len(tmesh)
        dx = xvqplmu[:nx*ntp].reshape((ntp, nx))
        dv = xvqplmu[nx*ntp:2*nx*ntp].reshape((ntp, nx))
        dq = xvqplmu[-2*nr*(ntp-1):-nr*(ntp-1)]
        dp = xvqplmu[-nr*(ntp-1):]
        xlist, vlist, plist = [xld[tmesh[0]]], [vld[tmesh[0]]], [pld[tmesh[0]]]
        for k, curt in enumerate(tmesh[1:]):
            xlist.append(xld[curt]+dx[k+1, :])
            vlist.append(vld[curt]+dv[k+1, :])
            plist.append(pld[curt]+dp[k])

        # getpdxdxg = get_pdxdxg(pld=pld, r=r)
        # getgmat = get_getgmat(xld=xld, holojaco=ovhdcrn['holojaco'])

        # getdualrhs = get_getdualrhs(cmat=cmat, qmat=qmat,
        #                             trgttrj=ystar, xld=xld)
        # tbmatt = rmatinv.dot(bmat.T)
        # terml = np.zeros((nx, 1))
        # gettermld = get_getdualrhs(cmat=cmat, qmat=smat,
        #                            trgttrj=ystar, xld=xld)
        # termld = -np.linalg.solve(mmat, gettermld(tmesh[-1]))

        # # tend = tmesh[-1]
        # # print xld[tend]
        # # print ystar(tend)
        # # ## make the terminal value consistent
        # curG = getgmat(tmesh[-1])
        # minvGt = np.dot(minv, curG.T)
        # csc = np.dot(curG, minvGt)
        # prjtermld = termld - \
        #     np.dot(minvGt, np.linalg.solve(csc, np.dot(curG, termld)))
        # # raise Warning('TODO: debug')

        # ulstt = bwsweep(tmesh=tmesh, amatfun=getbwamat, rhsfun=getdualrhs,
        #                 gmatfun=getgmat, mmat=mmat,
        #                 terml=terml, termld=prjtermld, outputmat=tbmatt)
        # uld = dict(zip(tmesh, ulstt))

        # def curinp(t):
        #     return uld[t].reshape((2, 1)) + keepitconst(t)

        # xlist, curulist, plist = \
        #     int_impeul_ggl(inix=inix, iniv=iniv,
        #                    # inpfun=testinp,
        #                    inpfun=curinp,
        #                    tmesh=tmesh, **ovhdcrn)
