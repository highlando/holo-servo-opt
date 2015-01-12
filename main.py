import numpy as np

import probdefs as pbd
import first_order_opti as fop
import seco_order_opti as sop
import plot_utils as plu

# parameters of the optimization problem
tE = 6.
Nts = 599
Riccati = False
udiril = [True, False]
bone = 1*1e-12
bzero = 1e-12
gamma = 0*1e-3
trjl = ['pwp', 'atan', 'plnm']
trgt = trjl[0]

# parameters of the target funcs
g0, gf = 0.5, 2.5
trnsarea = 1.  # size of the transition area in the pwl
polydeg = 5
tanpa = 8

# parameters of the system
defsysdict = dict(mvec=np.array([2., 1.]),
                  dvec=np.array([0.5]),
                  kvec=np.array([10.]),
                  printmats=True)
defprbdict = dict(posini=np.array([[0.5], [0]]),
                  velini=np.array([[0.], [0.]]))

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


if __name__ == '__main__':

    if Riccati:

        fbdict, ftdict = fop.solve_opt_ric(A=tA, B=tB, C=tC, tmesh=tmesh,
                                           gamma=gamma, beta=bzero,
                                           fpri=fpri, fdua=fdua, bt=tB.T)

        sysout, inpdict = fop.solve_cl_sys(A=tA, B=tB, C=tC,
                                           bmo=1./bzero, f=tf,
                                           tmesh=tmesh, zini=tini,
                                           fbd=fbdict, ftd=ftdict)

        plu.plot_output(tmesh, sysout, targetsig=trajec, inpdict=inpdict)

    else:
        def fo(t):
            return f[0]

        def ft(t):
            return f[1]

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

        leglist = ['x1', 'x2', 'l1', 'l2', 'u', 'x1-g']
        plotlist = [x1, x2, l1, l2, u, x1.flatten()-gvec]
        plu.plot_all(tmesh, plotlist, leglist=leglist)
