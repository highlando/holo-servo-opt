import numpy as np
import matplotlib.pyplot as plt

Nts = 100
tE = 6.
tM = tE/2
g0 = 0
gf = 1
tanpa = 2.
trnsarea = 1.

tmesh = np.linspace(0, tE, Nts)


# def trajec(t):
#     return 1 + g0 + np.tanh(tanpa*(t - tM))
# gvec = trajec(tmesh)

def trajec(t):
    if t < tM - trnsarea/2:
        print t, g0
        return g0
    elif t > tM + trnsarea/2:
        print t, gf
        return gf
    else:
        g = g0 + (gf-g0)*(t - tM + trnsarea/2)/trnsarea
        print t, g
        return g

gvec = np.zeros((Nts, 1))
for k, t in enumerate(tmesh):
    gvec[k] = trajec(t)


plt.figure(1111)
plt.plot(tmesh, gvec)
plt.show(block=False)
