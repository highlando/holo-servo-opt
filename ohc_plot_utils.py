import matplotlib.pyplot as plt
import numpy as np


def plt_inp(tmesh=None, exatinp=None):
    uflist, umlist = [], []
    for tk in tmesh:
        uflist.append(exatinp(tk)[0][0])
        umlist.append(exatinp(tk)[1][0])

    plt.figure(1)
    plt.plot(tmesh, uflist)
    plt.figure(2)
    plt.plot(tmesh, umlist)


def plttrjtrj(gfun=None, tmesh=None):
    ndts = 4
    xdsl, zdsl = [], []
    gt = gfun(tmesh[0])
    for k in range(ndts):
        xdsl.append([gt[k][0][0]])
        zdsl.append([gt[k][1][0]])
    for tk in tmesh[1:]:
        gt = gfun(tk)
        for k in range(ndts):
            xdsl[k].append(gt[k][0][0])
            zdsl[k].append(gt[k][1][0])
    for k in range(ndts):
        plt.figure(211)
        plt.plot(tmesh, xdsl[k])
    plt.figure(212)
    for k in range(ndts):
        plt.plot(tmesh, zdsl[k])


def plotxlist(xlist=None, tmesh=None):
    posarray = np.r_[xlist]
    plt.figure(123)
    plt.plot(posarray[:, 0], posarray[:, 1])
    if tmesh is not None:
        plt.figure(124)
        plt.plot(tmesh, posarray[:, 0])
        plt.figure(125)
        plt.plot(tmesh, posarray[:, 1])
