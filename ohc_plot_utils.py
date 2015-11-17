import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
    sns.set(style="whitegrid")
    mpilightgreen = '#BFDFDE'
    mpigraygreen = '#7DA9A8'
    # sns.set_palette(sns.dark_palette(mpigraygreen, 4, reverse=True))
    # sns.set_palette(sns.dark_palette(mpilightgreen, 6, reverse=True))
    # sns.set_palette('cool', 3)
    sns.set_palette('ocean_r', 7)
except ImportError:
    pass


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


def plot_ohc_xu(tmesh=None, xdlist=None, zdlist=None,
                uslist=None, ublist=None, betalist=None,
                tikzfile=None):
    plt.figure(989)
    for xdl in xdlist:
        plt.plot(tmesh, xdl)
    plt.figure(990)
    for xdl in zdlist:
        plt.plot(tmesh, xdl)
    plt.figure(991)
    for xdl in uslist:
        plt.plot(tmesh, xdl)
    plt.figure(992)
    for xdl in ublist:
        plt.plot(tmesh, xdl)
    plt.show(block=False)
