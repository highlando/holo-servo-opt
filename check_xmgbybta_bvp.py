import numpy as np

import probdefs as pbd
import seco_order_opti as sop

import matplotlib.pyplot as plt

# parameters of the optimization problem
tE = 6.
Nts = 599
Riccati = False
bone = 1e-12
gamma = 1e-14
bzerolist = [10**(-x) for x in np.arange(3, 14, 2)]
udiril = [True, True]

# which trajectory
trc = 3
trjl = ['pwl', 'atan', 'plnm', 'pwp']

# parameters of the target funcs
g0, gf = 0.5, 2.5
trnsarea = 5.  # size of the transition area in the pw polynomials
tanpa = 2
polydegl = [1, 3, 5, 7, 9]
polydeg = 9

# parameters of the system
# defsysdict = dict(mvec=np.array([2., 1.]),
#                   dvec=np.array([0.5]),
#                   kvec=np.array([10.]),
#                   printmats=True)
defsysdict = dict(mvec=np.array([2., 1., 1.]),
                  dvec=np.array([0.5, 0.5]),
                  kvec=np.array([10., 10]),
                  printmats=True)
defprbdict = dict(posini=np.array([[0.5], [0], [-0.5]]),
                  velini=np.array([[0.], [0.], [0.]]))

A, B, C, f = pbd.get_abcf(**defsysdict)
tmesh = pbd.get_tint(0.0, tE, Nts, sqzmesh=False, plotmesh=False)

fignum = 110
# for trc in range(1, 2):
#     trc = 1
for polydeg in polydegl:
    fignum = 1
    errl = []
    for bzero in bzerolist:
        trgt = trjl[trc]
        trajec = pbd.get_trajec(trgt,  tE=tE, g0=g0, gf=gf,
                                trnsarea=trnsarea, tanpa=tanpa,
                                polydeg=polydeg)
        gvec = np.zeros(tmesh.shape)
        for k, tc in enumerate(tmesh.tolist()):
            gvec[k] = trajec(tc)

        def fo(t):
            return f[0]

        def ft(t):
            return f[1]

        def ftt(t):
            return f[2]

        sol = sop.fd_fullsys(A=A, B=B, C=C, flist=[fo, ft, ftt], g=trajec,
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
        # plt.figure(1111)
        # plt.plot(tmesh, 1/bzero*abs(x1.flatten()-gvec))
        # plt.plot(tmesh, 1/bzero*abs(x2.flatten()-0*gvec))
        # plt.yscale('log')
        # plt.show(block=False)
        # errl.append(np.trapz(1/bzero*(x1.flatten()-gvec)**2, tmesh))
        errl.append(1/bzero*np.max(abs((x2.flatten()-gvec))))

    plt.figure(66)
    plt.plot(tmesh, gvec)
    plt.figure(fignum)
    plt.loglog(bzerolist, errl, 's', label='polydeg {0}'.format(polydeg))
    plt.legend()

plt.show(block=False)
