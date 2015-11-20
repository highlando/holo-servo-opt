import numpy as np
import scipy.optimize as sco

import holo_servo_opt.probdefs as pbd
import holo_servo_opt.ohc_utils as ocu

import holo_servo_opt.first_order_opti as foo
# import matplotlib.pyplot as plt
import matlibplots.conv_plot_utils as cpu


'''
the general model structure is

Mx'' = Ax - G.T(x)p + Bu + f
g(x) = 0
'''


def int_impeul_ggl(mmat=None, amat=None, rhs=None, holoc=None, holojaco=None,
                   bmat=None, inpfun=None, cmat=None,
                   inix=None, iniv=None, tmesh=None, retvlist=False, **kwargs):
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


# def get_getdgmat(xld=None, vld=None, holohess=None):
#     def getdgmat(t):
#         dxdxtg = holohess(xld[t])
#         curv = vld[t].reshape((nx, 1))
#         return np.dot(dxdxtg, curv).T
#     return getdgmat


def get_getdgmat(xld=None, vld=None, holohess=None):
    # dxdxtg = 2*np.array([[1, 0, -1, 0],
    #                      [0, -r**2, 0, 0],
    #                      [-1, 0, 1, 0],
    #                      [0, 0, 0, 1]])

    def getdgmat(t):
        dxdxtg = holohess(xld[t])
        curx = xld[t].reshape((nx, 1))
        return np.dot(dxdxtg, curx).T
    return getdgmat


# def get_getdualrhs(cmat=None, qmat=None, trgttrj=None, xld=None):
#     def getdualrhs(t):
#         curx = xld[t].reshape((nx, 1))
#         curey = trgttrj(t)
#         drhs = -np.dot(cmat.T, np.dot(qmat, np.dot(cmat, curx)-curey))
#         return drhs
#     return getdualrhs


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


def get_nwtncorr(xld=None, pld=None, holojaco=None, nx=None):
    def nwtncorr(t):
        return np.dot(holojaco(xld[t]).T, pld[t]).reshape((nx, 1))
    return nwtncorr


def getmddxld(vld=None, mmat=None, tmesh=None):
    mddxld = {}
    for k, curt in enumerate(tmesh[1:]):
        pret = tmesh[k]
        ctsi = 1./(curt - pret)
        mddxld.update({curt: ctsi*np.dot(mmat, vld[curt] - vld[pret])})

    return mddxld


def get_grhs(xld=None, holoc=None, holojaco=None):
    def grhs(t):
        return -(holoc(xld[t])-np.dot(holojaco(xld[t]), xld[t]))
    return grhs


def get_dgrhs(xld=None, vld=None, holojaco=None, holohess=None):
    def dgrhs(t):
        return -(-np.dot(np.dot(xld[t].T, holohess(xld[t])), vld[t]))
    return dgrhs


if __name__ == '__main__':
    tE, Nts = 6., 601
    tmesh = np.linspace(0, tE, Nts).tolist()
    # defining the target trajectory and the exact solution
    inix = np.array([[0, 40, 0, 4]]).T
    iniv = np.array([[0., 0., 0., 0.]]).T
    nx, ny, nu = inix.size, 2, 2
    nr = 1
    NP = 1
    # initial and final point
    gm0 = np.array([[0., 4.]]).T
    # gmf = np.array([[1., 5.]]).T
    # gmf = np.array([[0., 4.1]]).T
    gmf = np.array([[1., 5.]]).T
    # gmf = np.array([[1., 5.]]).T

    # scalar morphing function
    scalarg = pbd.get_trajec('pwp', tE=tE, g0=0., gf=1.,
                             trnsarea=tE, polydeg=13, retdts_all=True)
    # def of the problem
    modpardict = dict(J=.1, m=100., mt=10., r=.1, gravity=9.81)
    J, m, mt = modpardict['J'], modpardict['m'], modpardict['mt']
    r, gravity = modpardict['r'], modpardict['gravity']
    # whether to apply a constant momentum to counter the gravity
    counterg = True
    # def of the optimization problem
    qmat = np.eye(ny)
    beta = 1e-9
    betalist = [1e-5, 1e-6, 1e-7]
    legl = ['$\\beta = {0}\\quad$ '.format(bz) for bz in betalist]
    rmatinv = 1./beta*np.eye(nu)
    gamma = 1e-7
    smat = gamma*np.eye(ny)
    # the data of the problem
    ovhdcrn = ocu.overheadmodel(counterg=counterg,
                                **modpardict)
    mmat, cmat, bmat = ovhdcrn['mmat'], ovhdcrn['cmat'], ovhdcrn['bmat']
    minv = np.linalg.inv(mmat)
    exatinp = ocu.get_exatinp(scalarg=scalarg, gm0=gm0, gmf=gmf,
                              counterg=True, **modpardict)

    def ystar(t):
        return ocu.trgttrj(t, scalarg=scalarg,
                           gm0=gm0, gmf=gmf, retdts_all=False)
    ovhdcrn.update(dict(cmat=None))
    # the reference solution
    exdl, ezdl, eusl, eubl = [], [], [], []
    for t in tmesh:
        exdl.append(ystar(t)[0])
        ezdl.append(ystar(t)[1])
        eusl.append(exatinp(t)[0])
        eubl.append(exatinp(t)[1])
    bxdlist, bzdlist, buslist, bublist = [exdl], [ezdl], [eusl], [eubl]
    legl.insert(0, 'exact$\\quad$')  # Exakte L\\"osung')

    def zeroinp(t):
        return np.zeros((2, 1))

    def keepitconst(t):  # for constant position
        return np.array([[0, -m*gravity*r]]).T

    zxlist, zulist, zplist, zvlist = \
        int_impeul_ggl(inix=inix, iniv=iniv,
                       inpfun=zeroinp,
                       # inpfun=keepitconst,
                       # inpfun=exatinp,
                       tmesh=tmesh, retvlist=True, **ovhdcrn)
    xold = np.hstack(zxlist).reshape((Nts*nx, 1))
    zplist[0] = zplist[1]  # TODO: this is a hack for a consistent pini
    # raise Warning('TODO: debug')

    for cbeta in betalist:
        xlist, plist, vlist = zxlist, zplist, zvlist
        rmatinv = 1./cbeta*np.eye(nu)
        linsteps = 1
        for npc in range(linsteps):
            xld, pld = dict(zip(tmesh, xlist)), dict(zip(tmesh, plist))
            vld = dict(zip(tmesh, vlist))
            mddxld = getmddxld(vld=vld, mmat=mmat, tmesh=tmesh)

            getgmat = get_getgmat(xld=xld, holojaco=ovhdcrn['holojaco'])
            getdgmat = get_getdgmat(vld=vld, xld=xld,
                                    holohess=ovhdcrn['holohess'])
            getpdxdxg = get_pdxdxg(pld=pld, r=r)
            nwtncorr = get_nwtncorr(xld=xld, pld=pld, nx=nx,
                                    holojaco=ovhdcrn['holojaco'])

            xrhs = get_xresidual(xld=xld, pld=pld, sysrhs=ovhdcrn['rhs'],
                                 holojaco=ovhdcrn['holojaco'], mddxld=mddxld,
                                 minusres=True, nx=nx, NP=NP)
            grhs = get_grhs(xld=xld, holoc=ovhdcrn['holoc'],
                            holojaco=ovhdcrn['holojaco'])
            dgrhs = get_dgrhs(xld=xld, vld=vld, holojaco=ovhdcrn['holojaco'],
                              holohess=ovhdcrn['holohess'])

            def fwdrhs(t):
                return nwtncorr(t)+ovhdcrn['rhs']

            xvqpllmm = foo.\
                linoptsys_ltvgglholo(tmesh=tmesh, mmat=mmat, bmat=bmat,
                                     inpufun=None, getgmat=getgmat,
                                     getdgmat=getdgmat, getamat=getpdxdxg,
                                     xini=inix, vini=iniv, qmat=qmat,
                                     smat=smat, curterx=xld[tmesh[-1]],
                                     rmatinv=rmatinv, cmat=cmat, ystar=ystar,
                                     xrhs=xrhs, grhs=grhs, dgrhs=dgrhs, nr=nr)

            ntp = len(tmesh)
            ntpi = ntp-1
            dx = xvqpllmm[:nx*ntp].reshape((ntp, nx))
            dv = xvqpllmm[nx*ntp:2*nx*ntp].reshape((ntp, nx))
            dq = xvqpllmm[2*nx*ntp:2*nx*ntp+ntpi*nr]
            dp = xvqpllmm[2*nx*ntp+ntpi*nr:2*nx*ntp+2*ntpi*nr]
            nxvqp = 2*nx*ntp+2*ntpi*nr
            dl1 = xvqpllmm[nxvqp:nxvqp+nx*ntp].reshape((ntp, nx))
            dl2 = xvqpllmm[nxvqp+nx*ntp:nxvqp+2*nx*ntp].reshape((ntp, nx))
            dm1 = xvqpllmm[nxvqp+2*nx*ntp:nxvqp+2*nx*ntp+ntpi*nr]
            dm2 = xvqpllmm[nxvqp+2*nx*ntp+ntpi*nr:nxvqp+2*nx*ntp+2*ntpi*nr]
            xlist, vlist = [xld[tmesh[0]]], [vld[tmesh[0]]]
            # TODO p and q
            plist = [-dq[0]]  # [pld[tmesh[0]]]
            for k, curt in enumerate(tmesh[1:]):
                xlist.append(dx[k+1, :])
                vlist.append(dv[k+1, :])
                plist.append(-dq[k])

        bxdl, bzdl, busl, bubl = [], [], [], []
        rmbt = np.dot(rmatinv, bmat.T)
        for k, t in enumerate(tmesh):
            # bxdl.append(dx[k, 2])
            # bzdl.append(dx[k, 3])
            curu = np.dot(rmbt, dl2[k, :].T)
            busl.append(curu[0])
            bubl.append(curu[1])
        ulds, uldb = dict(zip(tmesh, busl)), dict(zip(tmesh, bubl))

        def optiinp(t):
            return np.array([[ulds[t], uldb[t]]]).T
        optxlist, optulist, optplist, optpvlist = \
            int_impeul_ggl(inix=inix, iniv=iniv,
                           inpfun=optiinp,
                           # inpfun=keepitconst,
                           # inpfun=exatinp,
                           tmesh=tmesh, retvlist=True, **ovhdcrn)
        for k, t in enumerate(tmesh):
            bxdl.append(optxlist[k][2])
            bzdl.append(optxlist[k][3])

        bxdlist.append(bxdl)
        bzdlist.append(bzdl)
        buslist.append(busl)
        bublist.append(bubl)

    cpu.para_plot(tmesh, bxdlist, leglist=legl, fignum=1,
                  xlabel='time $t~[s]$', ylabel='trajectory $x_d~[m]$',
                  tikzfile='ohc_xdtrajs.tikz',
                  title=None)  # 'Trajektorie')

    cpu.para_plot(tmesh, bzdlist, leglist=legl, fignum=2,
                  xlabel='time $t~[s]$', ylabel='trajectory $z_d~[m]$',
                  tikzfile='ohc_zdtrajs.tikz',
                  title=None)  # 'Trajektorie')

    cpu.para_plot(tmesh, buslist, leglist=legl, fignum=3,
                  xlabel='time $t~[s]$', ylabel='input $F~[N]$',
                  tikzfile='ohc_uss.tikz',
                  title=None)  # 'Trajektorie')

    cpu.para_plot(tmesh, bublist, leglist=legl, fignum=4,
                  xlabel='time $t~[s]$', ylabel='input $M_n~[Nm]$',
                  tikzfile='ohc_ubs.tikz',
                  title=None)  # 'Trajektorie')
    # plu.plot_ohc_xu(tmesh=tmesh, xdlist=bxdlist, zdlist=bzdlist,
    #                 uslist=buslist, ublist=bublist, betalist=betalist,
    #                 leglist=legl, tikzfile='tbd')
