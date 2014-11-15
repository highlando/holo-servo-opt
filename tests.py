import numpy as np
import main as mnf
import matplotlib.pyplot as plt


def solve_fbft_stst(A=None, bbt=None, ctc=None, fpri=None, fdua=None,
                    tmesh=None, bt=None, verbose=False):
    """solve for steadystate feedback gain and feedthrough

    """

    t = tmesh[-1]
    ststx = mnf.solve_algric(A=A, W=ctc, R=bbt, X0=0*bbt)

    rhs = np.dot(ststx, fpri) + fdua
    ststw = np.linalg.solve(-(A.T + np.dot(ststx, bbt)), rhs)

    fbdict = {t: np.dot(bt, ststx)}
    ftdict = {t: np.dot(bt, ststw)}

    for tk, t in reversed(list(enumerate(tmesh[:-1]))):
        fbdict.update({t: np.dot(bt, ststx)})
        ftdict.update({t: np.dot(bt, ststw)})

    return fbdict, ftdict


if __name__ == '__main__':
    steadystate = True
    N = 2
    tA = np.array([[-2., 1.], [1., -2.]])
    tini = np.zeros((N, 1))
    tB = np.ones((N, 1))
    tC = np.ones((N, 1)).T

    beta = 1e-2
    gamma = 1e-3

    tE = 5.
    Nts = 199
    tmesh = mnf.get_tint(0.0, tE, Nts, sqzmesh=False, plotmesh=False)

    bbt = 1./beta*np.dot(tB, tB.T)
    ctc = np.dot(tC.T, tC)

    g0 = np.dot(tC, tini)[0][0]
    gf = 1

    def trajec(t):
        trt = t/tE
        return g0 + (126*trt**5 - 420*trt**6 + 540*trt**7 -
                     315*trt**8 + 70*trt**9)*(gf - g0)

    def fpri(t):
        return np.zeros((N, 1))

    def fdua(t):
        return np.dot(tC.T, trajec(t))

    termw = gamma*fdua(tmesh[-1])
    termx = -gamma*ctc

    if steadystate:
        fbdict, ftdict = solve_fbft_stst(A=tA, bbt=bbt, ctc=ctc,
                                         fpri=fpri(0), fdua=fdua(tE),
                                         tmesh=tmesh, bt=tB.T)
    else:
        fbdict, ftdict = mnf.solve_fbft(A=tA, bbt=bbt, ctc=ctc,
                                        fpri=fpri, fdua=fdua, tmesh=tmesh,
                                        termx=termx, termw=termw, bt=tB.T)

    sysout = mnf.solve_cl_sys(A=tA, B=tB, C=tC, bmo=1./beta, f=fpri(0),
                              tmesh=tmesh, fbd=fbdict, ftd=ftdict, zini=tini)

    mnf.plot_output(tmesh, sysout, trajec)
    mnf.plot_fbft(tmesh, fbdict, ftdict)
    plt.show(block=False)
