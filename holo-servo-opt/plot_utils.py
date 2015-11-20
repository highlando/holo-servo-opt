import matplotlib.pyplot as plt
import numpy as np


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

    plt.show(block=False)


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
    plt.show(block=False)


def plot_xmgbybz(tmesh, outdict, targetsig, bzero):
    plt.figure(55)
    diffsigl = []
    for t in tmesh:
        diffsigl.append(1./bzero*(abs(outdict[t][0][0]-targetsig(t))))
    plt.plot(tmesh, diffsigl, label='$\\beta_0 = {0}$'.format(bzero))
    plt.yscale('log')
    plt.legend()
    plt.show(block=False)
