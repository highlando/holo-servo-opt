import numpy as np

import probdefs as pbd
import first_order_opti as fop
import plot_utils as plu

# parameters of the optimization problem
tE = 6.
Nts = 4799
Riccati = False
bzerolist = [10**(-x) for x in np.arange(6, 12)]
gamma = 0*1e-5
trc = 2


trjl = ['pwl', 'atan', 'plnm']

# parameters of the target funcs
g0, gf = 0.5, 2.5
trnsarea = .2  # size of the transition area in the pwl
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

for bzero in bzerolist:
    trgt = trjl[trc]
    trajec = pbd.get_trajec(trgt,  tE=tE, g0=g0, gf=gf,
                            trnsarea=trnsarea, tanpa=tanpa)
    gvec = np.zeros(tmesh.shape)
    for k, tc in enumerate(tmesh.tolist()):
        gvec[k] = trajec(tc)

    def fpri(t):
        return tf

    def fdua(t):
        return np.dot(tC.T, trajec(t))

    fbdict, ftdict = fop.solve_opt_ric(A=tA, B=tB, C=tC, tmesh=tmesh,
                                       gamma=gamma, beta=bzero,
                                       fpri=fpri, fdua=fdua, bt=tB.T)

    sysout, inpdict = fop.solve_cl_sys(A=tA, B=tB, C=tC,
                                       bmo=1./bzero, f=tf,
                                       tmesh=tmesh, zini=tini,
                                       fbd=fbdict, ftd=ftdict)

    plu.plot_xmgbybz(tmesh, sysout, trajec, bzero)
