import numpy as np

import probdefs as pbd
import first_order_opti as fop
import seco_order_opti as sop

import matlibplots.conv_plot_utils as cpu

# setup of the tests
opticheck, fbcheck = False, False
# make it come true
# opticheck = True
fbcheck = True

# parameters of the problem
ncar = 2

# parameters of the optimization problem
tE = 6.
Nts = 2400
udiril = [True, False]
bone = 0*1e-12
gamma = 1.  # 0*1e-3
trjl = ['pwp', 'atan', 'plnm']
trgt = trjl[0]

# parameters of the target funcs
g0, gf = 0.5, 2.5
trnsarea = 1.  # size of the transition area in the pwl
polydeg = 9
tanpa = 8

# parameters for the perturbation
peps = 0.3
veps = 0.3

# parameters of the system
if ncar == 2:
    defsysdict = dict(mvec=np.array([1., 2.]),
                      dvec=np.array([0.5]),
                      kvec=np.array([1.]),
                      printmats=True)
    defprbdict = dict(posini=np.array([[0.5], [0]]),
                      velini=np.array([[0.], [0.]]))
elif ncar == 3:
    defsysdict = dict(mvec=np.array([1., 1., 2.]),
                      dvec=np.array([0.5, 0.5]),
                      kvec=np.array([1., 1.]),
                      printmats=True)
    defprbdict = dict(posini=np.array([[0.5], [0], [-0.5]]),
                      velini=np.array([[0.], [0.], [0.]]))
else:
    raise NotImplementedError('only 2 or 3 cars')

A, B, C, f = pbd.get_abcf(**defsysdict)
tA, tB, tC, tf, tini = fop.comp_firstorder_mats(A=A, B=B, C=C, f=f,
                                                **defprbdict)

tmesh = pbd.get_tint(0.0, tE, Nts, sqzmesh=False, plotmesh=False)

trajec = pbd.get_trajec(trgt,  tE=tE, g0=g0, gf=gf, polydeg=polydeg,
                        trnsarea=trnsarea, tanpa=tanpa)

gvec = np.zeros(tmesh.shape)
for k, tc in enumerate(tmesh.tolist()):
    gvec[k] = trajec(tc)


def fpri(t):
    return tf


def fdua(t):
    return np.dot(tC.T, trajec(t))

dtrajec = pbd.get_trajec(trgt,  tE=tE, g0=g0, gf=gf, polydeg=polydeg,
                         trnsarea=trnsarea, tanpa=tanpa, retderivs=True)

# TODO: exact solution for 3 (or more) cars
fvec = np.zeros(tmesh.shape)
for k, tc in enumerate(tmesh.tolist()):
    fvec[k] = 2./defsysdict['kvec'][0]*dtrajec(tc)[2] + 1*dtrajec(tc)[1]


def forcefunc(t):
    return 2./defsysdict['kvec'][0]*dtrajec(t)[2] + 3*dtrajec(t)[1]

if __name__ == '__main__':
    bzerl = [10**(-x) for x in np.arange(5, 10, 2)]  # [10**(-7)]
    legl = ['$\\beta_0 = {0}$'.format(bz) for bz in bzerl]
    bzero = 1e-9

    if fbcheck:
        tinipv, tinipp, tinipppv = np.copy(tini), np.copy(tini), np.copy(tini)
        tinipp[0] = tini[0] + peps
        tinipv[ncar] = tini[ncar] + veps
        tinipv[ncar] = tini[ncar] + veps
        tinipppv[0] = tini[0] + peps
        tinipppv[ncar] = tini[ncar] + veps
        inilist = [tini, tinipv, tinipp, tinipppv]
        leglist = ['exact $x_{1,0}$, $\dot x_{1,0}$',
                   'perturbed  $\dot x_{1,0}$',
                   'perturbed  $x_{1,0}$',
                   'perturbed  $x_{1,0}$, $\dot x_{1,0}$']

        def inivinipcheck(inilist, fbd=None, ftd=None, bmo=None):
            outlist = []
            for inival in inilist:
                sysout, inpdict = fop.\
                    solve_cl_sys(A=tA, B=tB, C=tC, bmo=bmo, f=tf,
                                 tmesh=tmesh, zini=inival, fbd=fbd, ftd=ftd,
                                 retaslist=True)
                outlist.append(sysout)
            return outlist

        # Riccati solution
        fbdict, ftdict = fop.solve_opt_ric(A=tA, B=tB, C=tC, tmesh=tmesh,
                                           gamma=gamma, beta=bzero,
                                           fpri=fpri, fdua=fdua, bt=tB.T)

        outlist = inivinipcheck(inilist, fbd=fbdict, ftd=ftdict, bmo=1./bzero)

        ylims = [0., 3.5]
        cpu.para_plot(tmesh, outlist, leglist=leglist, fignum=4,
                      xlabel='time $t$', ylabel='trajectory $x_1$',
                      ylims=ylims,
                      tikzfile='{0}car_inivpert_Riccati.tikz'.format(ncar),
                      title=None, legloc='lower right')  # 'Trajektorie')

        # # direct solution
        outlist = inivinipcheck(inilist, fbd=None, ftd=forcefunc, bmo=1.)
        cpu.para_plot(tmesh, outlist, leglist=leglist, fignum=3,
                      xlabel='time $t$', ylabel='trajectory $x_1$',
                      ylims=ylims,
                      tikzfile='{0}car_inivpert_direct.tikz'.format(ncar),
                      title=None, legloc='lower right')  # 'Trajektorie')

    if opticheck:

        def fone(t):
            return f[0]

        def ftwo(t):
            return f[1]

        def fthr(t):
            return f[2]

        if ncar == 2:
            flist = [fone, ftwo]
        if ncar == 3:
            flist = [fone, ftwo, fthr]
        else:
            raise NotImplementedError('only 2 or 3 cars')

        ulist, xlist = [], []
        for bzero in bzerl:
            sol = sop.fd_fullsys(A=A, B=B, C=C, flist=flist, g=trajec,
                                 tmesh=tmesh.reshape((tmesh.size, 1)),
                                 Q=np.dot(C.T, C),
                                 bone=bone, bzero=bzero, gamma=gamma,
                                 inix=defprbdict['posini'],
                                 inidx=defprbdict['velini'],
                                 udiril=udiril)
            nT = tmesh.size
            x1 = sol[ncar*nT: (ncar+1)*nT]
            u = sol[-nT:]
            ulist.append(u)
            xlist.append(x1)

        ulist.insert(0, fvec)
        xlist.insert(0, gvec)
        legl.insert(0, 'exact')  # Exakte L\\"osung')
        print legl

        cpu.para_plot(tmesh, xlist, leglist=legl, fignum=3,
                      xlabel='time $t$', ylabel='trajectory $x_3$',
                      tikzfile='snapplot_{0}car_trajs.tikz'.format(ncar),
                      title=None)  # 'Trajektorie')

        cpu.para_plot(tmesh, ulist, leglist=legl, fignum=4,
                      xlabel='time $t$', ylabel='input $F$',
                      tikzfile='snapplot_{0}car_usex.tikz'.format(ncar),
                      title=None)  # 'Kontrollkraft')
