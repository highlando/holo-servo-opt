import numpy as np

import probdefs as pbd
import first_order_opti as fop
import seco_order_opti as sop
# import plot_utils as plu

import matlibplots.conv_plot_utils as cpu
# parameters of the problem
ncar = 2

# parameters of the optimization problem
tE = 6.
Nts = 599
udiril = [True, False]
bone = 0*1e-12
bzero = 1e-12
gamma = 1.  # 0*1e-3
trjl = ['pwp', 'atan', 'plnm']
trgt = trjl[0]

# parameters of the target funcs
g0, gf = 0.5, 2.5
trnsarea = 1.  # size of the transition area in the pwl
polydeg = 7
tanpa = 8

# parameters of the system
if ncar == 2:
    defsysdict = dict(mvec=np.array([1., 2.]),
                      dvec=np.array([0.5]),
                      kvec=np.array([1.]),
                      printmats=True)
    defprbdict = dict(posini=np.array([[0.5], [0]]),
                      velini=np.array([[0.], [0.]]))
elif ncar == 3:
    defsysdict = dict(mvec=np.array([1, 1., 2.]),
                      dvec=np.array([0.5, 0.5]),
                      kvec=np.array([1., 1.]),
                      printmats=True)
    defprbdict = dict(posini=np.array([[1], [0.5], [0]]),
                      velini=np.array([[0.], [0.], [0.]]))
else:
    raise NotImplementedError('only 2 or 3 cars')

A, B, C, f = pbd.get_abcf(**defsysdict)
tA, tB, tC, tf, tini = fop.comp_firstorder_mats(A=A, B=B, C=C, f=f,
                                                **defprbdict)

tmesh = pbd.get_tint(0.0, tE, Nts, sqzmesh=False, plotmesh=False)

trajec = pbd.get_trajec(trgt,  tE=tE, g0=g0, gf=gf, polydeg=polydeg,
                        trnsarea=trnsarea, tanpa=tanpa)

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

fvec = np.zeros(tmesh.shape)
for k, tc in enumerate(tmesh.tolist()):
    fvec[k] = 2./defsysdict['kvec'][0]*dtrajec(tc)[2] + 1*dtrajec(tc)[1]

if __name__ == '__main__':

    def fo(t):
        return f[0]

    def ft(t):
        return f[1]

    bzerl = [10**(-x) for x in np.arange(5, 10, 2)]
    legl = ['$\\beta_0 = {0}$'.format(bz) for bz in bzerl]

    ulist, xlist = [], []
    for bzero in bzerl:
        sol = sop.fd_fullsys(A=A, B=B, C=C, flist=[fo, ft], g=trajec,
                             tmesh=tmesh.reshape((tmesh.size, 1)),
                             Q=np.dot(C.T, C),
                             bone=bone, bzero=bzero, gamma=gamma,
                             inix=defprbdict['posini'],
                             inidx=defprbdict['velini'],
                             udiril=udiril)
        nT = tmesh.size
        l1 = sol[:nT]
        l2 = sol[nT:2*nT]
        x1 = sol[2*nT:3*nT]
        x2 = sol[3*nT:4*nT]
        u = sol[4*nT:]
        ulist.append(u)
        xlist.append(x1)

    # cpu.para_plot(tmesh, xlist, leglist=legl, fignum=1,
    #               xlabel='$t$', ylabel='$x_1$',
    #               tikzfile='snapplot_trajs.tikz',
    #               title='Optimal trajectory')
    # cpu.para_plot(tmesh, ulist, leglist=legl, fignum=2,
    #               xlabel='$t$', ylabel='$F$',
    #               tikzfile='snapplot_us.tikz',
    #               title='Optimal control force')

    ulist.insert(0, fvec)
    xlist.insert(0, gvec)
    legl.insert(0, 'Exakte L\\"o sung')
    print legl

    cpu.para_plot(tmesh, xlist, leglist=legl, fignum=3,
                  xlabel='$t$', ylabel='$x_2$',
                  tikzfile='snapplot_trajs.tikz',
                  title='Trajektorie')

    cpu.para_plot(tmesh, ulist, leglist=legl, fignum=4,
                  xlabel='$t$', ylabel='$F$',
                  tikzfile='snapplot_usex.tikz',
                  title='Kontrollkraft')
