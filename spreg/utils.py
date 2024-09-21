"""
Tools for different procedure estimations
"""

__author__ = "Luc Anselin lanselin@gmail.com, \
        Pedro V. Amaral pedro.amaral@asu.edu, \
        David C. Folch david.folch@asu.edu, \
        Daniel Arribas-Bel darribas@asu.edu,\
        Levi Wolf levi.john.wolf@gmail.com"

import numpy as np
from scipy import sparse as SP
import scipy.optimize as op
import numpy.linalg as la
from libpysal.weights.spatial_lag import lag_spatial
from libpysal.cg import KDTree        # new for make_wnslx
from scipy.sparse import coo_array,csr_array    # new for make_wnslx
from .sputils import *
import copy



class RegressionPropsY(object):

    """
    Helper class that adds common regression properties to any regression
    class that inherits it.  It takes no parameters.  See BaseOLS for example
    usage.

    Parameters
    ----------

    Attributes
    ----------
    mean_y  : float
              Mean of the dependent variable
    std_y   : float
              Standard deviation of the dependent variable

    """

    @property
    def mean_y(self):
        try:
            return self._cache["mean_y"]
        except AttributeError:
            self._cache = {}
            self._cache["mean_y"] = np.mean(self.y)
        except KeyError:
            self._cache["mean_y"] = np.mean(self.y)
        return self._cache["mean_y"]

    @mean_y.setter
    def mean_y(self, val):
        try:
            self._cache["mean_y"] = val
        except AttributeError:
            self._cache = {}
            self._cache["mean_y"] = val
        except KeyError:
            self._cache["mean_y"] = val

    @property
    def std_y(self):
        try:
            return self._cache["std_y"]
        except AttributeError:
            self._cache = {}
            self._cache["std_y"] = np.std(self.y, ddof=1)
        except KeyError:
            self._cache["std_y"] = np.std(self.y, ddof=1)
        return self._cache["std_y"]

    @std_y.setter
    def std_y(self, val):
        try:
            self._cache["std_y"] = val
        except AttributeError:
            self._cache = {}
            self._cache["std_y"] = val
        except KeyError:
            self._cache["std_y"] = val


class RegressionPropsVM(object):

    """
    Helper class that adds common regression properties to any regression
    class that inherits it.  It takes no parameters.  See BaseOLS for example
    usage.

    Parameters
    ----------

    Attributes
    ----------
    utu     : float
              Sum of the squared residuals
    sig2n    : float
              Sigma squared with n in the denominator
    sig2n_k : float
              Sigma squared with n-k in the denominator
    vm      : array
              Variance-covariance matrix (kxk)

    """

    @property
    def utu(self):
        try:
            return self._cache["utu"]
        except AttributeError:
            self._cache = {}
            self._cache["utu"] = np.sum(self.u ** 2)
        except KeyError:
            self._cache["utu"] = np.sum(self.u ** 2)
        return self._cache["utu"]

    @utu.setter
    def utu(self, val):
        try:
            self._cache["utu"] = val
        except AttributeError:
            self._cache = {}
            self._cache["utu"] = val
        except KeyError:
            self._cache["utu"] = val

    @property
    def sig2n(self):
        try:
            return self._cache["sig2n"]
        except AttributeError:
            self._cache = {}
            self._cache["sig2n"] = self.utu / self.n
        except KeyError:
            self._cache["sig2n"] = self.utu / self.n
        return self._cache["sig2n"]

    @sig2n.setter
    def sig2n(self, val):
        try:
            self._cache["sig2n"] = val
        except AttributeError:
            self._cache = {}
            self._cache["sig2n"] = val
        except KeyError:
            self._cache["sig2n"] = val

    @property
    def sig2n_k(self):
        try:
            return self._cache["sig2n_k"]
        except AttributeError:
            self._cache = {}
            self._cache["sig2n_k"] = self.utu / (self.n - self.k)
        except KeyError:
            self._cache["sig2n_k"] = self.utu / (self.n - self.k)
        return self._cache["sig2n_k"]

    @sig2n_k.setter
    def sig2n_k(self, val):
        try:
            self._cache["sig2n_k"] = val
        except AttributeError:
            self._cache = {}
            self._cache["sig2n_k"] = val
        except KeyError:
            self._cache["sig2n_k"] = val

    @property
    def vm(self):
        try:
            return self._cache["vm"]
        except AttributeError:
            self._cache = {}
            self._cache["vm"] = np.dot(self.sig2, self.xtxi)
        except KeyError:
            self._cache["vm"] = np.dot(self.sig2, self.xtxi)
        finally:
            return self._cache["vm"]

    @vm.setter
    def vm(self, val):
        try:
            self._cache["vm"] = val
        except AttributeError:
            self._cache = {}
            self._cache["vm"] = val
        except KeyError:
            self._cache["vm"] = val


def get_A1_het(S):
    """
    Builds A1 as in Arraiz et al :cite:`Arraiz2010`

    .. math::

        A_1 = W' W - diag(w'_{.i} w_{.i})

    ...

    Parameters
    ----------

    S               : csr_matrix
                      PySAL W object converted into Scipy sparse matrix

    Returns
    -------

    Implicit        : csr_matrix
                      A1 matrix in scipy sparse format

    """
    StS = S.T * S
    d = SP.spdiags([StS.diagonal()], [0], S.get_shape()[0], S.get_shape()[1])
    d = d.asformat("csr")
    return StS - d


def get_A1_hom(s, scalarKP=False):
    r"""
    Builds A1 for the spatial error GM estimation with homoscedasticity as in
    Drukker et al. [Drukker2011]_ (p. 9).

    .. math::

        A_1 = \{1 + [n^{-1} tr(W'W)]^2\}^{-1} \[W'W - n^{-1} tr(W'W) I\]

    ...

    Parameters
    ----------

    s               : csr_matrix
                      PySAL W object converted into Scipy sparse matrix
    scalarKP        : boolean
                      Flag to include scalar corresponding to the first moment
                      condition as in Drukker et al. [1]_ (Defaults to False)

    Returns
    -------

    Implicit        : csr_matrix
                      A1 matrix in scipy sparse format
    """
    n = float(s.shape[0])
    wpw = s.T * s
    twpw = np.sum(wpw.diagonal())
    e = SP.eye(n, n, format="csr")
    e.data = np.ones(int(n)) * (twpw / n)
    num = wpw - e
    if not scalarKP:
        return num
    else:
        den = 1.0 + (twpw / n) ** 2.0
        return num / den


def get_A2_hom(s):
    r"""
    Builds A2 for the spatial error GM estimation with homoscedasticity as in
    Anselin (2011) :cite:`Anselin2011`

    .. math::

        A_2 = \dfrac{(W + W')}{2}

    ...

    Parameters
    ----------
    s               : csr_matrix
                      PySAL W object converted into Scipy sparse matrix
    Returns
    -------
    Implicit        : csr_matrix
                      A2 matrix in scipy sparse format
    """
    return (s + s.T) / 2.0


def _moments2eqs(A1, s, u):
    """
    Helper to compute G and g in a system of two equations as in
    the heteroskedastic error models from Drukker et al. [Drukker2011]_
    ...

    Parameters
    ----------

    A1          : scipy.sparse.csr
                  A1 matrix as in the paper, different deppending on whether
                  it's homocedastic or heteroskedastic model

    s           : W.sparse
                  Sparse representation of spatial weights instance

    u           : array
                  Residuals. nx1 array assumed to be aligned with w

    Attributes
    ----------

    moments     : list
                  List of two arrays corresponding to the matrices 'G' and
                  'g', respectively.


    """
    n = float(s.shape[0])
    A1u = A1 * u
    wu = s * u
    g1 = np.dot(u.T, A1u)
    g2 = np.dot(u.T, wu)
    g = np.array([[g1][0][0], [g2][0][0]]) / n

    G11 = np.dot(u.T, ((A1 + A1.T) * wu))
    G12 = -np.dot((wu.T * A1), wu)
    G21 = np.dot(u.T, ((s + s.T) * wu))
    G22 = -np.dot(wu.T, (s * wu))
    G = np.array([[G11[0][0], G12[0][0]], [G21[0][0], G22[0][0]]]) / n
    return [G, g]


def optim_moments(moments_in, vcX=np.array([0]), all_par=False, start=None, hard_bound=False):
    """
    Optimization of moments
    ...

    Parameters
    ----------

    moments     : Moments
                  Instance of gmm_utils.moments_het with G and g
    vcX         : array
                  Optional. Array with the Variance-Covariance matrix to be used as
                  weights in the optimization (applies Cholesky
                  decomposition). Set empty by default.
    all_par     : boolean
                  Optional. Whether to return all parameters from
                  solution or just the 1st. Default is 1st only.
    start       : list
                  List with initial values for the optimization
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
                   
    Returns
    -------
    x, f, d     : tuple
                  x -- position of the minimum
                  f -- value of func at the minimum
                  d -- dictionary of information from routine
                        d['warnflag'] is
                            0 if converged
                            1 if too many function evaluations
                            2 if stopped for another reason, given in d['task']
                        d['grad'] is the gradient at the minimum (should be 0 ish)
                        d['funcalls'] is the number of function calls made
    """
    moments = copy.deepcopy(moments_in)
    if vcX.any():
        Ec = np.transpose(la.cholesky(la.inv(vcX)))
        moments[0] = np.dot(Ec, moments_in[0])
        moments[1] = np.dot(Ec, moments_in[1])
    scale = np.min([[np.min(moments[0]), np.min(moments[1])]])
    moments[0], moments[1] = moments[0] / scale, moments[1] / scale
    if moments[0].shape[0] == 2:
        optim_par = lambda par: foptim_par(
            np.array([[float(par[0]), float(par[0]) ** 2.0]]).T, moments
        )
        start = [0.0]
        bounds = [(-0.99, 0.99)]
    if moments[0].shape[0] == 3:
        optim_par = lambda par: foptim_par(
            np.array([[float(par[0]), float(par[0]) ** 2.0, float(par[1])]]).T, moments
        )
        start = [0.0, 1.0]
        bounds = [(-0.99, 0.99), (0.0, None)]
    if moments[0].shape[1] == 4:
        optim_par = lambda par: foptim_par(
            np.array(
                [[float(par[0]), float(par[0]) ** 2.0, float(par[1]), float(par[2])]]
            ).T,
            moments,
        )
        if not start:
            start = [0.0, 1.0, 1.0]
        bounds = [(-0.99, 0.99), (0.0, None), (0.0, None)]
    lambdaX = op.fmin_l_bfgs_b(optim_par, start, approx_grad=True, bounds=bounds)

    if hard_bound:
        if abs(lambdaX[0][0]) >= 0.99:
            raise Exception("Spatial parameter was outside the bounds of -0.99 and 0.99")

    if all_par:
        return lambdaX[0]
    return lambdaX[0][0]


def foptim_par(par, moments):
    """
    Preparation of the function of moments for minimization
    ...

    Parameters
    ----------

    lambdapar       : float
                      Spatial autoregressive parameter
    moments         : list
                      List of Moments with G (moments[0]) and g (moments[1])

    Returns
    -------

    minimum         : float
                      sum of square residuals (e) of the equation system
                      moments.g - moments.G * lambdapar = e
    """
    vv = np.dot(moments[0], par)
    vv2 = moments[1] - vv
    return sum(vv2 ** 2)


def get_spFilter(w, lamb, sf):
    """
    Compute the spatially filtered variables

    Parameters
    ----------
    w       : weight
              PySAL weights instance
    lamb    : double
              spatial autoregressive parameter
    sf      : array
              the variable needed to compute the filter

    Returns
    --------
    rs      : array
              spatially filtered variable

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import get_spFilter
    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> w=libpysal.io.open(libpysal.examples.get_path("columbus.gal")).read()
    >>> solu = get_spFilter(w,0.5,y)
    >>> print(solu[0:5])
    [[  -8.9882875]
     [ -20.5685065]
     [ -28.196721 ]
     [ -36.9051915]
     [-111.1298   ]]

    """
    try:
        ws = w.sparse
    except:
        ws = w
    T = sf.shape[0] // ws.shape[0]
    if T == 1:
        result = sf - lamb * (ws * sf)
    else:
        result = sf - lamb * SP.kron(SP.identity(T), ws).dot(sf)
    return result


def get_lags(w, x, w_lags):
    """
    Calculates a given order of spatial lags and all the smaller orders

    Parameters
    ----------
    w       : weight
              PySAL weights instance
    x       : array
              nxk arrays with the variables to be lagged
    w_lags  : integer
              Maximum order of spatial lag

    Returns
    --------
    rs      : array
              nxk*(w_lags) array with spatially lagged variables

    """
    lag = lag_spatial(w, x)
    spat_lags = lag
    for i in range(w_lags - 1):
        lag = lag_spatial(w, lag)
        spat_lags = sphstack(spat_lags, lag)
    return spat_lags

def get_lags_split(w, x, max_lags, split_at):
    """
    Calculates a given order of spatial lags and all the smaller orders,
    separated into two groups (up to split_at and above)

    Parameters
    ----------
    w       : weight
              PySAL weights instance
    x       : array
              nxk arrays with the variables to be lagged
    max_lags  : integer
              Maximum order of spatial lag
    split_at: integer
              Separates the resulting lags into two cc: up to split_at and above

    Returns
    --------
    rs_l,rs_h: tuple of arrays
               rs_l: nxk*(split_at) array with spatially lagged variables up to split_at
               rs_h: nxk*(w_lags-split_at) array with spatially lagged variables above split_at

    """
    rs_l = lag = lag_spatial(w, x)
    rs_h = None
    if 0 < split_at < max_lags:
        for _ in range(split_at-1):
            lag = lag_spatial(w, lag)
            rs_l = sphstack(rs_l, lag)

        for i in range(max_lags - split_at):
            lag = lag_spatial(w, lag)
            rs_h = sphstack(rs_h, lag) if i > 0 else lag
    else:
        raise ValueError("max_lags must be greater than split_at and split_at must be greater than 0")

    return rs_l, rs_h

def inverse_prod(
    w,
    data,
    scalar,
    post_multiply=False,
    inv_method="power_exp",
    threshold=0.0000000001,
    max_iterations=None,
):
    """

    Parameters
    ----------

    w               : Pysal W object
                      nxn Pysal spatial weights object

    data            : Numpy array
                      nx1 vector of data

    scalar          : float
                      Scalar value (typically rho or lambda)

    post_multiply   : boolean
                      If True then post-multiplies the data vector by the
                      inverse of the spatial filter, if false then
                      pre-multiplies.
    inv_method      : string
                      If "true_inv" uses the true inverse of W (slow);
                      If "power_exp" uses the power expansion method (default)

    threshold       : float
                      Test value to stop the iterations. Test is against
                      sqrt(increment' * increment), where increment is a
                      vector representing the contribution from each
                      iteration.

    max_iterations  : integer
                      Maximum number of iterations for the expansion.

    Examples
    --------

    >>> import numpy, libpysal
    >>> import numpy.linalg as la
    >>> from spreg import inverse_prod
    >>> np.random.seed(10)
    >>> w = libpysal.weights.util.lat2W(5, 5)
    >>> w.transform = 'r'
    >>> data = np.random.randn(w.n)
    >>> data.shape = (w.n, 1)
    >>> rho = 0.4
    >>> inv_pow = inverse_prod(w, data, rho, inv_method="power_exp")

    # true matrix inverse

    >>> inv_reg = inverse_prod(w, data, rho, inv_method="true_inv")
    >>> np.allclose(inv_pow, inv_reg, atol=0.0001)
    True
    >>> # test the transpose version
    >>> inv_pow = inverse_prod(w, data, rho, inv_method="power_exp", post_multiply=True)
    >>> inv_reg = inverse_prod(w, data, rho, inv_method="true_inv", post_multiply=True)
    >>> np.allclose(inv_pow, inv_reg, atol=0.0001)
    True

    """
    if inv_method == "power_exp":
        inv_prod = power_expansion(
            w,
            data,
            scalar,
            post_multiply=post_multiply,
            threshold=threshold,
            max_iterations=max_iterations,
        )
    elif inv_method == "true_inv":
        try:
            matrix = la.inv(np.eye(w.n) - (scalar * w.full()[0]))
        except:
            matrix = la.inv(np.eye(w.shape[0]) - (scalar * w))
        if post_multiply:
#            inv_prod = spdot(data.T, matrix)
            inv_prod = np.matmul(data.T,matrix)   # inverse matrix is dense, wrong type in spdot
        else:
#            inv_prod = spdot(matrix, data)
            inv_prod = np.matmul(matrix,data)
    else:
        raise Exception("Invalid method selected for inversion.")
    return inv_prod


def power_expansion(
    w, data, scalar, post_multiply=False, threshold=0.0000000001, max_iterations=None
):
    r"""
    Compute the inverse of a matrix using the power expansion (Leontief
    expansion).  General form is:

        .. math:: 
            x &= (I - \rho W)^{-1}v = [I + \rho W + \rho^2 WW + \dots]v \\
              &= v + \rho Wv + \rho^2 WWv + \dots

    Examples
    --------
    Tests for this function are in inverse_prod()

    """
    try:
        ws = w.sparse
    except:
        ws = w
    if post_multiply:
        data = data.T
    running_total = copy.copy(data)
    increment = copy.copy(data)
    count = 1
    test = 10000000
    if max_iterations == None:
        max_iterations = 10000000
    while test > threshold and count <= max_iterations:
        if post_multiply:
            increment = increment * ws * scalar
        else:
            increment = ws * increment * scalar
        running_total += increment
        test_old = test
        test = la.norm(increment)
        if test > test_old:
            raise Exception(
                "power expansion will not converge, check model specification and that weight are less than 1"
            )
        count += 1
    return running_total


def set_endog(y, x, w, yend, q, w_lags, lag_q, slx_lags=0,slx_vars="All"):
    # Create spatial lag of y
    yl = lag_spatial(w, y)
    # spatial and non-spatial instruments
    if issubclass(type(yend), np.ndarray):
        if slx_lags > 0:
            lag_x, lag_xq = get_lags_split(w, x, slx_lags+1, slx_lags)
        else:
            lag_xq = x
        if lag_q:
            lag_vars = sphstack(lag_xq, q)
        else:
            lag_vars = lag_xq
        spatial_inst = get_lags(w, lag_vars, w_lags)
        q = sphstack(q, spatial_inst)
        yend = sphstack(yend, yl)
    elif yend == None:  # spatial instruments only
        if slx_lags > 0:
            lag_x, lag_xq = get_lags_split(w, x, slx_lags+w_lags, slx_lags)
        else:
            lag_xq = get_lags(w, x, w_lags)
        q = lag_xq
        yend = yl
    else:
        raise Exception("invalid value passed to yend")
    if slx_lags == 0:
        return yend, q
    else:  # ajdust returned lag_x here using slx_vars
        if (isinstance(slx_vars,list)):     # slx_vars has True,False
            if len(slx_vars) != x.shape[1] :
                raise Exception("slx_vars incompatible with x column dimensions")
            else:  # use slx_vars to extract proper columns
                vv = slx_vars * slx_lags
                lag_x = lag_x[:,vv]
            return yend, q, lag_x
        else:  # slx_vars is "All"
            return yend, q, lag_x



def set_endog_sparse(y, x, w, yend, q, w_lags, lag_q):
    """
    Same as set_endog, but with a sparse object passed as weights instead of W object.
    """
    yl = w * y
    # spatial and non-spatial instruments
    if issubclass(type(yend), np.ndarray):
        if lag_q:
            lag_vars = sphstack(x, q)
        else:
            lag_vars = x
        spatial_inst = w * lag_vars
        for i in range(w_lags - 1):
            spatial_inst = sphstack(spatial_inst, w * spatial_inst)
        q = sphstack(q, spatial_inst)
        yend = sphstack(yend, yl)
    elif yend == None:  # spatial instruments only
        q = w * x
        for i in range(w_lags - 1):
            q = sphstack(q, w * q)
        yend = yl
    else:
        raise Exception("invalid value passed to yend")
    return yend, q


def iter_msg(iteration, max_iter):
    if iteration == max_iter:
        iter_stop = "Maximum number of iterations reached."
    else:
        iter_stop = "Convergence threshold (epsilon) reached."
    return iter_stop


def sp_att(w, y, predy, w_y, rho, hard_bound=False):
    xb = predy - rho * w_y
    if np.abs(rho) < 1:
        predy_sp = inverse_prod(w, xb, rho)
        warn = None
        # Note 1: Here if omitting pseudo-R2; If not, see Note 2.
        resid_sp = y - predy_sp
    else:
        if hard_bound:
            raise Exception(
                "Spatial autoregressive parameter is outside the maximum/minimum bounds."
            )
        else:
            # warn = "Warning: Estimate for rho is outside the boundary (-1, 1). Computation of true inverse of W was required (slow)."
            # predy_sp = inverse_prod(w, xb, rho, inv_method="true_inv")
            warn = "*** WARNING: Estimate for spatial lag coefficient is outside the boundary (-1, 1). ***"
            predy_sp = np.zeros(y.shape, float)
            resid_sp = np.zeros(y.shape, float)
    # resid_sp = y - predy_sp #Note 2: Here if computing true inverse; If not,
    # see Note 1.
    return predy_sp, resid_sp, warn


def set_warn(reg, warn):
    """Groups warning messages for printout."""
    if warn:
        try:
            reg.warning += "Warning: " + warn + "\n"
        except:
            reg.warning = "Warning: " + warn + "\n"
    else:
        pass


def RegressionProps_basic(
    reg, betas=None, predy=None, u=None, sig2=None, sig2n_k=None, vm=None
):
    """Set props based on arguments passed."""
    if betas is not None:
        reg.betas = betas
    if predy is not None:
        reg.predy = predy
    else:
        try:
            reg.predy = spdot(reg.z, reg.betas)
        except:
            reg.predy = spdot(reg.x, reg.betas)
    if u is not None:
        reg.u = u
    else:
        reg.u = reg.y - reg.predy
    if sig2 is not None:
        reg.sig2 = sig2
    elif sig2n_k:
        reg.sig2 = np.sum(reg.u ** 2) / (reg.n - reg.k)
    else:
        reg.sig2 = np.sum(reg.u ** 2) / reg.n
    if vm is not None:
        reg.vm = vm

def optim_k(trace, window_size=None):
    """
    Finds optimal number of regimes for the endogenous spatial regimes model
    using a method adapted from Mojena (1977)'s Rule Two.

    Parameters
    ----------
    trace      : list
                    List of SSR values for different number of regimes
    window_size : integer
                    Size of the window to be used in the moving average
                    (Defaults to N//4)
    Returns
    -------
    i+window_size : integer
                    Optimal number of regimes

    Examples
    --------
    >>> import libpysal as ps
    >>> import numpy as np
    >>> import spreg
    >>> data = ps.io.open(ps.examples.get_path('NAT.dbf'))
    >>> y = np.array(data.by_col('HR90')).reshape((-1,1))
    >>> x_var = ['PS90','UE90']
    >>> x = np.array([data.by_col(name) for name in x_var]).T
    >>> w = ps.weights.Queen.from_shapefile(ps.examples.get_path("NAT.shp"))
    >>> x_std = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
    >>> reg = spreg.Skater_reg().fit(20, w, x_std, {'reg':spreg.OLS,'y':y,'x':x}, quorum=100, trace=True)
    >>> spreg.optim_k([reg._trace[i][1][2] for i in range(1, len(reg._trace))])
    9


    """

    N = len(trace)
    if not window_size:
        window_size = N//4 # Mojena suggests from 70% to 90%
        if window_size < 2:
            window_size = N
    std_dev = [np.std(trace[i:i+window_size]) for i in range(N - window_size + 1)]
    ma = np.convolve(trace, np.ones(window_size)/window_size, mode='valid')
    treshold = [True]
    i = 0
    while treshold[-1] and i < (N - window_size):
        b = (6/(window_size*(window_size*window_size-1))
            )*((2*np.sum(np.arange(1, i+2)*trace[window_size-1:i+window_size])
            )-((window_size+1)*np.sum(trace[window_size-1:i+window_size])))
        l = (window_size-1)*b/2
        treshold.append(trace[i+window_size] < ma[i] - b - l - 2.75*std_dev[i])
        i += 1
    return i+window_size


def make_wnslx(coords,params,leafsize=30,distance_metric='Euclidean'):
    '''
    
    Computes transformed distances as triangular kernel weights for transform = 'power', or fraction of maximum distance 
    (bandwidth) for transform = 'exponential'. Uses libpysal.cg.KDTree. The three main characteristics of the kernel weights
    are passed as a tuple with k (number of nearest neighbors), upper_distance_bound (variable or fixed bandwidth),
    and transform (power or exponential).

    With distance_upper_bound=np.inf, the computation is for a variable bandwidth, where the bandwidth 
    is determined by the number of nearest neighbors, k. When a distance_upper_bound is set that is larger than the 
    largest k-nearest neighbor distance, there is no effect. When the distance_upper_bound is less than the max 
    k-nearest neighbor distance for a given point, then it has the effect of imposing a fixed bandwidth, and 
    truncating the number of nearest neighbors to those within the bandwidth. As a result, the number of neighbors 
    will be less than k.

    Note that k is a binding constraint, so if imposing the bandwidth is important, k should be taken large enough.

    Parameters
    ----------
    coords               : n by 2 numpy array of x,y coordinates
    params               : tuple with
       k                 : number of nearest neighbors (the diagonal is not included in k, so to obtain
                           k real nearest neighbors KDTree query must be called with k+1
       distance_upper_bound : bandwidth (see above for interpretation), np.inf is for no bandwidth, used by
                           KDTree query
       transform         : determines type of transformation, triangular kernel weights for 'power',
                           fractional distance for 'exponential' (exponential)
    leafsize             : argument to construct KDTree, default is 30 (from sklearn)
    distance_metric      : type of distance, default is "Euclidean", other option is "Arc" for arc-distance, to be used with long,lat
                           (note: long should be x and lat is y), both are supported by libpysal.cg.KDTree, but not
                           by its scipy and sklearn counterparts


    Returns
    -------
    spdis                : transformed distance matrix as CSR sparse array

    '''
    k = params[0]
    distance_upper_bound = params[1]
    transform = params[2]
    kdt = KDTree(coords,leafsize=leafsize,distance_metric=distance_metric)
    dis,nbrs = kdt.query(coords,k=k+1,distance_upper_bound=distance_upper_bound) 
    # get rid of diagonals
    dis = dis[:,1:]
    nbrs = nbrs[:,1:]
    n = dis.shape[0]

    # maximum distance in each row
    if (np.isinf(distance_upper_bound)): # no fixed bandwidth
        mxrow = dis[:,-1].reshape(-1,1)
    else:
        dis = np.nan_to_num(dis,copy=True,posinf=0)   # turn inf to zero
        mxrow = np.amax(dis,axis=1).reshape(-1,1)

    # rescaled distance
    fdis = dis / mxrow

    if transform.lower() == 'power':   # triangular kernel weights
        fdis = -fdis + 1.0
    elif transform.lower() == 'exponential':   # distance fraction
        fdis = fdis
    else:
        raise Exception("Method not supported")

    # turn into COO sparse format and then into CSR
    kk = fdis.shape[1]
    rowids = np.repeat(np.arange(n),kk)
    if (np.isinf(distance_upper_bound)):
        colids = nbrs.flatten()
        fdis = fdis.flatten()
    else: # neighbors outside bandwidth have ID n
        pickgd = (nbrs != n)
        rowids = rowids[pickgd.flatten()]
        colids = nbrs[pickgd].flatten()
        fdis = fdis[pickgd].flatten()
    
    spdis = coo_array((fdis,(rowids,colids)))
    spdis = spdis.tocsr(copy=True)
    
    return spdis


def _test():
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    _test()
