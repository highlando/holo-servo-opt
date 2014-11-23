import numpy as np
import matplotlib.pyplot as plt

import first_order_opti as fop
import seco_order_opti as sop

tE = 6.
Nts = 599
g0, gf = 0.5, 2.5
Riccati = True
udiril = [True, False]
bone = 1e-8
bzero = 1e-6
gamma = 0*1e-5
trjl = ['pwl', 'atan', 'plnm']
trgt = trjl[0]

# parameters of the target funcs
trnsarea = .4  # size of the transition area in the pwl


def plot_output(tmesh, outdict, inpdict=None, targetsig=None):
    plt.figure(44)
    outsigl = []
    trgsigl = []
    diffsigl = []
    for t in tmesh:
        outsigl.append(outdict[t][0][0])
        trgsigl.append(targetsig(t))
        diffsigl.append(outsigl[-1] - trgsigl[-1])

    plt.subplot(311)
    plt.plot(tmesh, outsigl, label='output')
    plt.plot(tmesh, trgsigl, label='target trajec')
    plt.legend(loc=4)
    plt.subplot(312)
    plt.plot(tmesh, diffsigl, label='diff in output')
    plt.legend()

    if inpdict is not None:
        inpl = []
        for t in tmesh:
            inpl.append(np.linalg.norm(inpdict[t]))
        plt.subplot(313)
        plt.plot(tmesh, inpl, label='$\|Bu\|$')
        plt.legend()


def plot_fbft(tmesh, fbdict, ftdict):
    normfbl = []
    ftl = []
    for t in tmesh:
        normfbl.append(np.linalg.norm(fbdict[t]))
        ftl.append(ftdict[t][0])

    plt.figure(11)
    plt.plot(tmesh, normfbl)
    plt.figure(22)
    plt.plot(tmesh, ftl)


def get_tint(t0, tE, Nts, sqzmesh=True, plotmesh=False):
    """set up a (nonuniform) time mesh

    for tracking problems a mesh with nodes clustered at the ends
    is advisable since

     1. the backward in time Riccati problem needs a high resolution towards \
    the terminal value
     2. the closed loop problem has large gradients at the beginning

    this function uses the sin function to squeeze a mesh towards the \
    marginal points
    """
    if sqzmesh:
        taux = np.linspace(-0.5*np.pi, 0.5*np.pi, Nts+1)
        taux = (np.sin(taux) + 1)*0.5  # squeeze and adjust to [0, 1]
        tint = (t0 + (tE-t0)*taux).flatten()  # adjust to [t0, tE]
    else:
        tint = np.linspace(t0, tE, Nts+1).flatten()
    if plotmesh:
        import matplotlib.pyplot as plt
        plt.plot(tint, np.ones(Nts+1), '.')
        plt.show()

    return tint


def get_abcf(kvec=None, mvec=None, dvec=None, N=None, printmats=False):
    """system matrices for the multi mass servo problem

    `F -> mN <-> mN-1 <-> .... <-> m1`

    Parameters
    ----------
    kvec : (N-1, ) array, optional
       the spring constants
    mvec : (N, ) array, optional
       the masses
    dvec : (N, ) array, optional
        the spring offsets
    N : integer, optional
        if provided, the matrices for a system of size `N`
        with all parameters equal to `1` is returned
    printmats : boolean, optional
        whether to print the matrices to the screen, defaults to false

    Returns
    -------
    A : (N, N) array
        coefficient matrix like `kvec/mvec*delta_x`
    B : (N, 1) array
        `(0, 0, ..., 1).T`
    C : (1, N) array
        `(1, 0, ..., 0)`
    f : (N, 1) array
        inhomogeneity, basically `kvec/mvec*dvec`
    """
    if N is None:
        N = mvec.size
    else:
        kvec = np.ones((N-1, 1))
        dvec = np.ones((N-1, 1))
        mvec = np.ones((N, 1))

    A = np.zeros((N, N))
    f = np.zeros((N, 1))
    B = np.zeros((N, 1))
    C = np.zeros((1, N))

    difftonext = np.array([[1, -1]])

    # eqn for first mass body m1
    k = 0
    A[k, k:k+2] = -kvec[k]/mvec[k]*difftonext
    C[0, k] = 1
    f[k] = kvec[k]/mvec[k]*dvec[k]

    # eqns for the inner mass bodies
    for k in range(1, N-1):
        # consider the next body
        A[k, k:k+2] = -kvec[k]/mvec[k]*difftonext
        f[k] += kvec[k]/mvec[k]*dvec[k]
        # consider the previous body
        A[k, k-1: k+1] = A[k, k-1: k+1] + kvec[k-1]/mvec[k]*difftonext
        f[k] += -kvec[k-1]/mvec[k]*dvec[k-1]

    # eqn for last mass body
    k = N-1
    A[k, k-1: k+1] = kvec[k-1]/mvec[k]*difftonext
    f[k] = -kvec[k-1]/mvec[k]*dvec[k-1]
    B[k, 0] = 1
    if printmats:
        print '\nA=\n', A
        print '\nB=\n', B
        print '\nC=\n', C
        print '\nf=\n', f

    return A, B, C, f

if __name__ == '__main__':
    defsysdict = dict(mvec=np.array([2., 1.]),
                      dvec=np.array([0.5]),
                      kvec=np.array([10.]),
                      printmats=True)
    defprbdict = dict(posini=np.array([[0.5], [0]]),
                      velini=np.array([[0.], [0.]]))

    if trgt == 'pwl':
        tM = tE/2

        def trajec(t):
            if t < tM - trnsarea/2:
                g = g0
            elif t > tM + trnsarea/2:
                g = gf
            else:
                g = g0 + (gf-g0)*(t - tM + trnsarea/2)/trnsarea
            return g

    elif trgt == 'atan':
        tM = tE/2
        tanpa = 8

        def trajec(t):
            return 1 + g0 + np.tanh(tanpa*(t - tM))

    elif trgt == 'plnm':

        def trajec(t):
            trt = t/tE
            return g0 + (126*trt**5 - 420*trt**6 + 540*trt**7 -
                         315*trt**8 + 70*trt**9)*(gf - g0)

    defctrldict = dict(gamma=1e-3,
                       beta=1e-3,
                       g=trajec)
    A, B, C, f = get_abcf(**defsysdict)
    tA, tB, tC, tf, tini = fop.comp_firstorder_mats(A=A, B=B, C=C, f=f,
                                                    **defprbdict)
    tmesh = get_tint(0.0, tE, Nts, sqzmesh=False, plotmesh=False)

    beta, gamma = defctrldict['beta'], defctrldict['gamma']

    def fpri(t):
        return tf

    def fdua(t):
        return np.dot(tC.T, trajec(t))

    if Riccati:
        bbt = 1./beta*np.dot(tB, tB.T)
        ctc = np.dot(tC.T, tC)
        termw = gamma*fdua(tmesh[-1])
        termx = -gamma*ctc

        fbdict, ftdict = fop.solve_fbft(A=tA, bbt=bbt, ctc=ctc,
                                        fpri=fpri, fdua=fdua, tmesh=tmesh,
                                        termx=termx, termw=termw, bt=tB.T)

        sysout, inpdict = fop.solve_cl_sys(A=tA, B=tB, C=tC, bmo=1./beta, f=tf,
                                           tmesh=tmesh, fbd=fbdict, ftd=ftdict,
                                           zini=tini)

        plot_output(tmesh, sysout, targetsig=trajec, inpdict=inpdict)
        # plot_fbft(tmesh, fbdict, ftdict)
        plt.show(block=False)

    else:
        def fo(t):
            return f[0]

        def ft(t):
            return f[1]

        sol, gvec \
            = sop.fd_fullsys(A=A, B=B, C=C, flist=[fo, ft], g=trajec,
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

        def plot_all(tmesh, pltlist, leglist=None):
            plt.figure(111)
            spl = [321, 322, 323, 324, 325, 326]
            for k, data in enumerate(pltlist):
                plt.subplot(spl[k])
                plt.plot(tmesh, data)
                if leglist is not None:
                    plt.title(leglist[k])
            plt.show(block=False)
            return

        if trgt == 'pwl':
            gvec = np.zeros(tmesh.shape)
            for k, tc in enumerate(tmesh.tolist()):
                gvec[k] = trajec(tc)

        else:
            gvec = trajec(tmesh)

        leglist = ['x1', 'x2', 'l1', 'l2', 'u', 'x1-g']
        plotlist = [x1, x2, l1, l2, u, x1.flatten()-gvec]
        plot_all(tmesh, plotlist, leglist=leglist)
