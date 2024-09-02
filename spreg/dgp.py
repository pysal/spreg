"""
Generating spatial models for specific DGP

"""

__author__ = "Luc Anselin lanselin@gmail.com,\
     Pedro Amaral pedrovma@gmail.com,\
     Renan Serenini renan.serenini@uniroma1.it"

import numpy as np
import math
import libpysal
from scipy.linalg import expm
from .utils import inverse_prod


__all__ = [
    "make_error",
    "make_x",
    "make_wx",
    "make_xb",
    "make_wxg",
    "dgp_errproc",
    "dgp_ols",
    "dgp_slx",
    "dgp_sperror",
    "dgp_slxerror",
    "dgp_lag",
    "dgp_spdurbin",
    "dgp_lagerr",
    "dgp_gns",
    "dgp_mess",
]


def make_error(rng, n, mu=0, varu=1, method="normal"):
    """
    make_error: generate error term for a given distribution

    Arguments:
    ----------
    rng:      random number object
    n:        number of observations
    mu:       mean (when needed)
    varu:     variance (when needed)
    method:   type of distribution, one of
              normal, laplace, cauchy, lognormal

    Returns:
    --------
    u:        nx1 vector of random errors

    Examples
    --------

    >>> import numpy as np
    >>> from spreg import make_error
    >>> rng = np.random.default_rng(12345)
    >>> make_error(rng,5)
    array([[-1.42382504],
           [ 1.26372846],
           [-0.87066174],
           [-0.25917323],
           [-0.07534331]])

    """
    # normal - standard normal is default
    if method == "normal":
        sdu = math.sqrt(varu)
        u = rng.normal(loc=mu, scale=sdu, size=n).reshape(n, 1)
    # laplace with thicker tails
    elif method == "laplace":
        sdu = math.sqrt(varu / 2.0)
        u = rng.laplace(loc=mu, scale=sdu, size=n).reshape(n, 1)
    # cauchy, ill-behaved, no mean or variance defined
    elif method == "cauchy":
        u = rng.standard_cauchy(size=n).reshape(n, 1)
    elif method == "lognormal":
        sdu = math.sqrt(varu)
        u = rng.lognormal(mean=mu, sigma=sdu, size=n).reshape(n, 1)
    # all other yield warning
    else:
        print("Warning: Unsupported distribution")
        u = None
    return u


def make_x(rng, n, mu=[0], varu=[1], cor=0, method="uniform"):
    """
    make_x: generate a matrix of k columns of x for a given distribution

    Arguments:
    ----------
    rng:      random number object
    n:        number of observations
    mu:       mean as a list
    varu:     variance as a list
    cor:      correlation as a float (for bivariate normal only)
    method:   type of distribution, one of
              uniform, normal, bivnormal (bivariate normal)

    Returns:
    --------
    x:        nxk matrix of x variables

    Note:
    -----
    Uniform and normal generate separate draws, bivariate normal generates
    correlated draws

    Examples
    --------

    >>> import numpy as np
    >>> from spreg import make_x
    >>> rng = np.random.default_rng(12345)
    >>> make_x(rng,5,mu=[0,1],varu=[1,4])
    array([[0.78751508, 2.30580253],
           [1.09728308, 4.14520464],
           [2.76215497, 1.29373239],
           [2.3426149 , 4.6609906 ],
           [1.35484323, 6.52500165]])

    """
    # check on k dimension
    k = len(mu)
    if k == len(varu):
        # initialize
        x = np.zeros((n, k))
        for i in range(k):
            # uniform - range is derived from variance since var = (1/12)range^2
            # range is found as square root of 12 times variance
            # for 0-1, varu should be 0.0833333
            # low is always 0
            if method == "uniform":
                sdu = math.sqrt(12.0 * varu[i])
                x[:, i] = rng.uniform(low=0, high=sdu, size=n)
            # normal - independent normal draws
            elif method == "normal":
                sdu = math.sqrt(varu[i])
                x[:, i] = rng.normal(loc=mu[i], scale=sdu, size=n)
            # bivariate normal - only for k=2
            elif method == "bivnormal":
                if k != 2:
                    print("Error: Wrong dimension for k")
                    x = None
                    return x
                else:
                    ucov = cor * math.sqrt(varu[0] * varu[1])
                    mcov = [[varu[0], ucov], [ucov, varu[1]]]
                    x = rng.multivariate_normal(mean=mu, cov=mcov, size=n)
                    return x
            else:
                print("Warning: Unsupported distribution")
                x = None
    else:
        x = None
    return x


def make_wx(x, w, o=1):
    """
    make_wx: generate a matrix spatially lagged x given matrix x

             x must be previously generated using make_x, no constant included

    Arguments:
    ----------
    x:        x matrix - no constant
    w:        row-standardized spatial weights in spreg format
    o:        order of contiguity, default o=1

    Returns:
    --------
    wx:       nx(kxo) matrix of spatially lagged x variables

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, make_wx
    >>> rng = np.random.default_rng(12345)
    >>> x = make_x(rng,25,mu=[0],varu=[1])
    >>> w  = libpysal.weights.lat2W(5, 5)
    >>> w.transform = "r"
    >>> make_wx(x,w)[0:5,:]
    array([[1.12509217],
           [1.87409079],
           [1.36225472],
           [2.1491645 ],
           [2.80255786]])

    """
    if w.n != x.shape[0]:
        print("Error: incompatible weights dimensions")
        return None
    w1x = libpysal.weights.lag_spatial(w, x)
    wx = w1x
    if o > 1:
        for i in range(1, o):
            whx = libpysal.weights.lag_spatial(w, w1x)
            w1x = whx
            wx = np.hstack((wx, whx))
    return wx


def make_xb(x, beta):
    """
    make_xb: generate a column xb as matrix x (constant added)
             times list beta (includes coefficient for constant term)

    Arguments:
    ----------
    x:        n x (k-1) matrix for x variables
    beta:     k length list of regression coefficients

    Returns:
    --------
    xb:        nx1 vector of x times beta

    Examples
    --------

    >>> import numpy as np
    >>> from spreg import make_x, make_xb
    >>> rng = np.random.default_rng(12345)
    >>> x = make_x(rng,5,mu=[0,1],varu=[1,4])
    >>> make_xb(x,[1,2,3])
    array([[ 9.49243776],
           [15.63018007],
           [10.4055071 ],
           [19.66820159],
           [23.28469141]])
    """
    n = x.shape[0]
    k = x.shape[1]
    if k + 1 != len(beta):
        print("Error: Incompatible dimensions")
        return None
    else:
        b = np.array(beta)[:, np.newaxis]
        x1 = np.hstack((np.ones((n, 1)), x))  # include constant
        xb = np.dot(x1, b)
        return xb


def make_wxg(wx, gamma):
    """
    make_wxg: generate a column wxg as matrix wx (no constant)
             times list gamma (coefficient of spatially lagged x)

    Arguments:
    ----------
    wx:       n x ((k-1)xo) matrix for spatially lagged x variables of all orders
    gamma:    (k-1)*o length list of regression coefficients for spatially lagged x

    Returns:
    --------
    wxg:      nx1 vector of wx times gamma

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, make_wx, make_wxg
    >>> rng = np.random.default_rng(12345)
    >>> x = make_x(rng,25,mu=[0],varu=[1])
    >>> w  = libpysal.weights.lat2W(5, 5)
    >>> w.transform = "r"
    >>> wx = make_wx(x,w)
    >>> print(wx.shape)
    (25, 1)
    >>> make_wxg(wx,[2,4])[0:5,:]
    array([[ 2.25018434,  4.50036868],
           [ 3.74818158,  7.49636316],
           [ 2.72450944,  5.44901889],
           [ 4.298329  ,  8.59665799],
           [ 5.60511572, 11.21023145]])

    """
    k = wx.shape[1]
    if k > 1:
        if k != len(gamma):
            print("Error: Incompatible dimensions")
            return None
        else:
            g = np.array(gamma)[:, np.newaxis]
            wxg = np.dot(wx, g)
    else:  # gamma is a scalar
        wxg = wx * gamma
    return wxg


def dgp_errproc(u, w, lam=0.5, model="sar", imethod="power_exp"):
    """
    dgp_errproc: generates pure spatial error process

    Arguments:
    ----------
    u:      random error vector
    w:      spatial weights object
    lam:    spatial autoregressive parameter
    model:  type of process ('sar' or 'ma')
    imethod: method for inverse transformation, default = 'power_exp'

    Returns:
    --------
    y : vector of observations following a spatial AR or MA process

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, dgp_errproc
    >>> rng = np.random.default_rng(12345)
    >>> u = make_x(rng,25,mu=[0],varu=[1], method='normal')
    >>> w  = libpysal.weights.lat2W(5, 5)
    >>> w.transform = "r"
    >>> dgp_errproc(u,w)[0:5,:]
    array([[-1.43760658],
           [ 0.69778271],
           [-0.7750646 ],
           [-0.47750452],
           [-0.72377417]])

    """
    n0 = u.shape[0]
    if w.n != n0:
        print("Error: incompatible weights dimensions")
        return None
    if model == "sar":
        y = inverse_prod(w, u, lam, inv_method=imethod)
    elif model == "ma":
        y = u + lam * libpysal.weights.lag_spatial(w, u)
    else:
        print("Error: unsupported model type")
        return None
    return y


def dgp_ols(u, xb):
    """
    dgp_ols: generates y for non-spatial process with given xb and error term u

    Arguments:
    ----------
    u:      random error vector
    xb:     vector of xb

    Returns:
    ----------
    y: vector of observations on dependent variable

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, dgp_ols
    >>> rng = np.random.default_rng(12345)
    >>> u = make_x(rng,25,mu=[0],varu=[1], method='normal')
    >>> x = make_x(rng,25,mu=[0],varu=[1])
    >>> xb = make_xb(x,[1,2])
    >>> dgp_ols(u,xb)[0:5,:]
    array([[5.22803968],
           [3.60291127],
           [1.02632633],
           [1.37589879],
           [5.07165754]])

    """
    n1 = u.shape[0]
    n2 = xb.shape[0]
    if n1 != n2:
        print("Error: dimension mismatch")
        return None
    y = xb + u
    return y


def dgp_slx(u, xb, wxg):
    """
    dgp_slx: generates y for SLX with given xb, wxg, and error term u

    Arguments:
    ----------
    u:      random error vector
    xb:     vector of xb
    wxg:    vector of wxg

    Returns:
    ----------
    y: vector of observations on dependent variable

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, make_xb, make_wx, make_wxg, dgp_slx
    >>> rng = np.random.default_rng(12345)
    >>> u = make_x(rng,25,mu=[0],varu=[1], method='normal')
    >>> x = make_x(rng,25,mu=[0],varu=[1])
    >>> xb = make_xb(x,[1,2])
    >>> w  = libpysal.weights.lat2W(5, 5)
    >>> w.transform = "r"
    >>> wx = make_wx(x,w)
    >>> wxg = make_wxg(wx,[2])
    >>> dgp_slx(u, xb, wxg)[0:5,:]
    array([[8.85854389],
           [7.17524694],
           [3.83674621],
           [4.73103929],
           [8.37023076]])
    """
    n0 = u.shape[0]
    n1 = xb.shape[0]
    n2 = wxg.shape[0]
    if n0 != n1:
        print("Error: dimension mismatch")
        return None
    elif n1 != n2:
        print("Error: dimension mismatch")
        return None
    y = xb + wxg + u
    return y


def dgp_sperror(u, xb, w, lam=0.5, model="sar", imethod="power_exp"):
    """
    dgp_sperror: generates y for spatial error model with given xb, weights,
                  spatial parameter lam, error term, method for inverse transform

    Arguments:
    ----------
    u:       random error
    xb:      vector of xb
    w:       spatial weights
    lam:     spatial coefficient
    model:   type of process ('sar' or 'ma')
    imethod: method for inverse transformation, default = 'power_exp'

    Returns:
    ----------
    y: vector of observations on dependent variable

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, make_xb, dgp_sperror
    >>> rng = np.random.default_rng(12345)
    >>> u = make_x(rng,25,mu=[0],varu=[1], method='normal')
    >>> x = make_x(rng,25,mu=[0],varu=[1])
    >>> xb = make_xb(x,[1,2])
    >>> w  = libpysal.weights.lat2W(5, 5)
    >>> w.transform = "r"
    >>> dgp_sperror(u, xb, w)[0:5,:]
    array([[5.21425813],
           [3.03696553],
           [1.12192347],
           [1.15756751],
           [4.42322667]])
    """
    n0 = u.shape[0]
    n1 = xb.shape[0]
    if n0 != n1:
        print("Error: incompatible weights dimensions")
        return None
    elif w.n != n1:
        print("Error: incompatible weights dimensions")
        return None
    if model == "sar":
        u1 = inverse_prod(w, u, lam, inv_method=imethod)
    elif model == "ma":
        u1 = u + lam * libpysal.weights.lag_spatial(w, u)
    else:
        print("Error: unsupported model type")
        return None
    y = xb + u1
    return y


def dgp_slxerror(u, xb, wxg, w, lam=0.5, model="sar", imethod="power_exp"):
    """
    dgp_sperror: generates y for SLX spatial error model with xb, wxg, weights,
                  spatial parameter lam, model type (sar or ma),
                  error term, method for inverse transform

    Arguments:
    ----------
    u:       random error
    xb:      vector of xb
    wxg:     vector of wxg
    w:       spatial weights
    lam:     spatial coefficient
    model:   type of process ('sar' or 'ma')
    imethod: method for inverse transformation, default = 'power_exp'

    Returns:
    ----------
    y: vector of observations on dependent variable

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, make_xb, make_wx, make_wxg, dgp_slxerror
    >>> rng = np.random.default_rng(12345)
    >>> u = make_x(rng,25,mu=[0],varu=[1], method='normal')
    >>> x = make_x(rng,25,mu=[0],varu=[1])
    >>> xb = make_xb(x,[1,2])
    >>> w  = libpysal.weights.lat2W(5, 5)
    >>> w.transform = "r"
    >>> wx = make_wx(x,w)
    >>> wxg = make_wxg(wx,[2])
    >>> dgp_slxerror(u,xb,wxg,w)[0:5,:]
    array([[8.84476235],
           [6.6093012 ],
           [3.93234334],
           [4.51270801],
           [7.7217999 ]])
    """

    n0 = u.shape[0]
    n1 = xb.shape[0]
    n2 = wxg.shape[0]
    if n0 != n1:
        print("Error: dimension mismatch")
        return None
    elif n1 != n2:
        print("Error: dimension mismatch")
        return None
    if w.n != n1:
        print("Error: incompatible weights dimensions")
        return None
    if model == "sar":
        u1 = inverse_prod(w, u, lam, inv_method=imethod)
    elif model == "ma":
        u1 = u + lam * libpysal.weights.lag_spatial(w, u)
    else:
        print("Error: unsupported model type")
        return None
    y = xb + wxg + u1
    return y


def dgp_lag(u, xb, w, rho=0.5, imethod="power_exp"):
    """
    dgp_lag:      generates y for spatial lag model with xb, weights,
                  spatial parameter rho,
                  error term, method for inverse transform

    Arguments:
    ----------
    u:       random error
    xb:      vector of xb
    w:       spatial weights
    rho:     spatial coefficient
    imethod: method for inverse transformation, default = 'power_exp'

    Returns:
    ----------
    y: vector of observations on dependent variable

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, make_xb, dgp_lag
    >>> rng = np.random.default_rng(12345)
    >>> u = make_x(rng,25,mu=[0],varu=[1], method='normal')
    >>> x = make_x(rng,25,mu=[0],varu=[1])
    >>> xb = make_xb(x,[1,2])
    >>> w  = libpysal.weights.lat2W(5, 5)
    >>> w.transform = "r"
    >>> dgp_lag(u, xb, w)[0:5,:]
    array([[10.16582326],
           [ 7.75359864],
           [ 5.39821733],
           [ 5.62244672],
           [ 8.868168  ]])
    """
    n0 = u.shape[0]
    n1 = xb.shape[0]
    if n0 != n1:
        print("Error: dimension mismatch")
        return None
    if w.n != n1:
        print("Error: incompatible weights dimensions")
        return None
    y1 = xb + u
    y = inverse_prod(w, y1, rho, inv_method=imethod)
    return y


def dgp_spdurbin(u, xb, wxg, w, rho=0.5, imethod="power_exp"):
    """
    dgp_spdurbin: generates y for spatial Durbin model with xb, wxg, weights,
                  spatial parameter rho,
                  error term, method for inverse transform

    Arguments:
    ----------
    u:       random error
    xb:      vector of xb
    wxg:     vector of wxg
    w:       spatial weights
    rho:     spatial coefficient
    imethod: method for inverse transformation, default = 'power_exp'

    Returns:
    ----------
    y: vector of observations on dependent variable

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, make_xb, make_wx, make_wxg, dgp_spdurbin
    >>> rng = np.random.default_rng(12345)
    >>> u = make_x(rng,25,mu=[0],varu=[1], method='normal')
    >>> x = make_x(rng,25,mu=[0],varu=[1])
    >>> xb = make_xb(x,[1,2])
    >>> w  = libpysal.weights.lat2W(5, 5)
    >>> w.transform = "r"
    >>> wx = make_wx(x,w)
    >>> wxg = make_wxg(wx,[2])
    >>> dgp_spdurbin(u,xb,wxg,w)[0:5,:]
    array([[18.06895353],
           [15.18686487],
           [11.95080505],
           [12.55220513],
           [15.75805066]])
    """
    n0 = u.shape[0]
    n1 = xb.shape[0]
    n2 = wxg.shape[0]
    if n0 != n1:
        print("Error: dimension mismatch")
        return None
    elif n1 != n2:
        print("Error: dimension mismatch")
        return None
    if w.n != n1:
        print("Error: incompatible weights dimensions")
    y1 = xb + wxg + u
    y = inverse_prod(w, y1, rho, inv_method=imethod)
    return y


def dgp_lagerr(u, xb, w, rho=0.5, lam=0.2, model="sar", imethod="power_exp"):
    """
    dgp_lagerr:   generates y for spatial lag model with sar or ma errors
                  with xb, weights,
                  spatial parameter rho, spatial parameter lambda,
                  model for spatial process,
                  error term, method for inverse transform

    Arguments:
    ----------
    u:       random error
    xb:      vector of xb
    w:       spatial weights
    rho:     spatial coefficient for lag
    lam:     spatial coefficient for error
    model:   spatial process for error
    imethod: method for inverse transformation, default = 'power_exp'

    Returns:
    ----------
    y: vector of observations on dependent variable

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, make_xb, dgp_lagerr
    >>> rng = np.random.default_rng(12345)
    >>> u = make_x(rng,25,mu=[0],varu=[1], method='normal')
    >>> x = make_x(rng,25,mu=[0],varu=[1])
    >>> xb = make_xb(x,[1,2])
    >>> w  = libpysal.weights.lat2W(5, 5)
    >>> w.transform = "r"
    >>> dgp_lagerr(u, xb, w)[0:5,:]
    array([[10.13845523],
           [ 7.53009531],
           [ 5.40644034],
           [ 5.51132886],
           [ 8.58872366]])
    """
    n0 = u.shape[0]
    n1 = xb.shape[0]
    if n0 != n1:
        print("Error: dimension mismatch")
        return None
    if w.n != n1:
        print("Error: incompatible weights dimensions")
        return None
    if model == "sar":
        u1 = inverse_prod(w, u, lam, inv_method=imethod)
    elif model == "ma":
        u1 = u + lam * libpysal.weights.lag_spatial(w, u)
    else:
        print("Error: unsupported model type")
        return None
    y1 = xb + u1
    y = inverse_prod(w, y1, rho, inv_method=imethod)
    return y


def dgp_gns(u, xb, wxg, w, rho=0.5, lam=0.2, model="sar", imethod="power_exp"):
    """
    dgp_gns:      generates y for general nested model with sar or ma errors
                  with xb, wxg, weights,
                  spatial parameter rho, spatial parameter lambda,
                  model for spatial process,
                  error term, method for inverse transform

    Arguments:
    ----------
    u:       random error
    xb:      vector of xb
    wxg:     vector of wxg
    w:       spatial weights
    rho:     spatial coefficient for lag
    lam:     spatial coefficient for error
    model:   spatial process for error
    imethod: method for inverse transformation, default = 'power_exp'

    Returns:
    ----------
    y: vector of observations on dependent variable

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, make_xb, make_wx, make_wxg, dgp_gns
    >>> rng = np.random.default_rng(12345)
    >>> u = make_x(rng,25,mu=[0],varu=[1], method='normal')
    >>> x = make_x(rng,25,mu=[0],varu=[1])
    >>> xb = make_xb(x,[1,2])
    >>> w  = libpysal.weights.lat2W(5, 5)
    >>> w.transform = "r"
    >>> wx = make_wx(x,w)
    >>> wxg = make_wxg(wx,[2])
    >>> dgp_gns(u,xb,wxg,w)[0:5,:]
    array([[18.04158549],
           [14.96336153],
           [11.95902806],
           [12.44108728],
           [15.47860632]])
    """
    n0 = u.shape[0]
    n1 = xb.shape[0]
    n2 = wxg.shape[0]
    if n0 != n1:
        print("Error: dimension mismatch")
        return None
    elif n1 != n2:
        print("Error: dimension mismatch")
        return None
    if w.n != n1:
        print("Error: incompatible weights dimensions")
    if model == "sar":
        u1 = inverse_prod(w, u, lam, inv_method=imethod)
    elif model == "ma":
        u1 = u + lam * libpysal.weights.lag_spatial(w, u)
    else:
        print("Error: unsupported model type")
        return None
    y1 = xb + wxg + u1
    y = inverse_prod(w, y1, rho, inv_method=imethod)
    return y


def dgp_mess(u, xb, w, rho=0.5):
    """
    dgp_mess:     generates y for MESS spatial lag model with xb, weights,
                  spatial parameter rho (gets converted into alpha),
                  sigma/method for the error term

    Arguments:
    ----------
    u:       random error
    xb:      vector of xb
    w:       spatial weights
    rho:     spatial coefficient (converted into alpha)

    Returns:
    ----------
    y: vector of observations on dependent variable

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import make_x, make_xb, dgp_mess
    >>> rng = np.random.default_rng(12345)
    >>> u = make_x(rng,25,mu=[0],varu=[1], method='normal')
    >>> x = make_x(rng,25,mu=[0],varu=[1])
    >>> xb = make_xb(x,[1,2])
    >>> w  = libpysal.weights.lat2W(5, 5)
    >>> w.transform = "r"
    >>> dgp_mess(u, xb, w)[0:5,:]
    array([[10.12104421],
           [ 7.45561055],
           [ 5.32807674],
           [ 5.55549492],
           [ 8.62685145]])
    """
    n0 = u.shape[0]
    n1 = xb.shape[0]
    if n0 != n1:
        print("Error: dimension mismatch")
        return None
    if w.n != n1:
        print("Error: incompatible weights dimensions")
        return None
    bigw = libpysal.weights.full(w)[0]
    alpha = np.log(1 - rho)  # convert between rho and alpha
    aw = -alpha * bigw  # inverse exponential is -alpha
    xbu = xb + u
    y = np.dot(expm(aw), xbu)
    return y


def _test():
    import doctest

    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == "__main__":
    _test()
