import numpy as np
# import scipy.optimize as sco

# import probdefs as pbd
# import matplotlib.pyplot as plt


def overheadmodel(J=None, m=None, mt=None, r=None, gravity=9.81):
    """ provide the model of the overhead crane

    described by the state `x` with

     * `x[0]:= s` - the x position of the cart
     * `x[1]:= beta` - modelling how pulling of the rope
     * `x[2]:= xd` - the x-position of the load
     * `x[3]:= zd` - the y-position of the load

    Returns
    ---
    ovhdcrn : dict
        with the following keys
         * `mmat`: the mass matrix
         * `amat`: the stiffness matrix
         * `bmat`: the input matrix
         * `holoc`: callable `g(x)` ret. the value of the constraint at `x`
         * `holojaco`: callable `G(x)` returning the Jacobi matrix at `x`

    Examples
    ---
    ovhdcrn = overheadmodel(J=.1, m=100., mt=10., r=.1)
    """

    mmat = np.diag([mt, J, m, m])

    # state: x = [s, beta, xd, zd].T

    amat = np.zeros((4, 4))
    bmat = np.array([[1., 0], [0, 1.], [0, 0], [0, 0]])
    cmat = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])

    rhs = np.array([[0, 0, 0, m*gravity]]).T

    def holoc(x=None):
        return (x[2] - x[0])**2 + x[3]**2 - (r*x[1])**2

    def holojaco(x):
        return 2*np.array([[-(x[2]-x[0]), -r**2*x[1], x[2]-x[0], x[3]]])

    ovhdcrn = dict(mmat=mmat, amat=amat, bmat=bmat, cmat=cmat,
                   rhs=rhs, holoc=holoc, holojaco=holojaco)
    return ovhdcrn


def trgttrj(t, scalarg=None, gm0=None, gmf=None, retdts_all=True):
    sclrgt = scalarg(t)  # , retdts_all=retdts_all)
    if retdts_all:
        return (gm0 + sclrgt[0]*(gmf - gm0),
                + sclrgt[1]*(gmf - gm0),
                + sclrgt[2]*(gmf - gm0),
                + sclrgt[3]*(gmf - gm0),
                + sclrgt[4]*(gmf - gm0))
    else:
        return gm0 - sclrgt[0]*(gmf - gm0)


def get_exatinp(m=None, r=None, mt=None, J=None, gravity=None):
    def exatinp(t):
        def _auxvals(curt):
            ctg = trgttrj(curt, retdts_all=True)
            g1, g2 = ctg[0][0], ctg[0][1]
            d1g1, d2g1, d3g1, d4g1 = ctg[1][0], ctg[2][0], ctg[3][0], ctg[4][0]
            d1g2, d2g2, d3g2, d4g2 = ctg[1][1], ctg[2][1], ctg[3][1], ctg[4][1]

            # define aux variables as in rob Alt, Betsch, Yang '14 w/ exact sol
            gmd2g2 = gravity - d2g2
            lmbd = .5*m/g2*gmd2g2
            s = g1 + d2g1*g2/gmd2g2
            bt = + g2/(r*gmd2g2)*np.sqrt(d2g1**2+gmd2g2**2)
            tt = d1g1 + d2g1*g2*d3g2/gmd2g2**2 + (d3g1*g2+d2g1*d1g2)/gmd2g2
            alpha = (d2g1*tt - d2g1*d1g1 + d1g2*gmd2g2) /\
                (r*np.sqrt(d2g1**2+gmd2g2**2))
            th = d2g1 + 2*d2g1*g2*d3g2**2/gmd2g2**3 +\
                (2*d3g1*g2*d3g2+2*d2g1*d1g2*d3g2+d2g1*g2*d4g2)/gmd2g2**2 +\
                (d4g1*g2+2*d3g1*d1g2+d2g1*d2g2)/gmd2g2
            halp = ((s-g1)*th + (g1-s)*d2g1 + g2*d2g2 + (tt-d1g1)*tt -
                    r**2*alpha**2 + (d1g1-tt)*d1g1 + d1g2**2)/(r**2*bt)
            return th, lmbd, halp, bt, s, g1
        th, lmbd, halp, bt, s, g1 = _auxvals(t)
        uF = mt*th + 2*(s-g1)*lmbd
        uM = J*halp - 2*r**2*bt*lmbd

        return np.array([uF, uM])
    return exatinp
