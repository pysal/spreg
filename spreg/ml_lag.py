"""
ML Estimation of Spatial Lag Model
"""

__author__ = "Luc Anselin lanselin@gmail.com, \
              Pedro V. Amaral pedrovma@gmail.com, \
              Serge Rey srey@asu.edu, \
              Levi Wolf levi.john.wolf@gmail.com"

import numpy as np
import numpy.linalg as la
from scipy import sparse as sp
from scipy.sparse.linalg import splu as SuperLU
from .utils import RegressionPropsY, RegressionPropsVM, inverse_prod, set_warn, get_lags
from .sputils import spdot, spfill_diagonal, spinv
from . import diagnostics as DIAG
from . import user_output as USER
import pandas as pd
from .output import output, _nonspat_top, _spat_diag_out, _spat_pseudo_r2, _summary_impacts
from .w_utils import symmetrize
from libpysal import weights

try:
    from scipy.optimize import minimize_scalar

    minimize_scalar_available = True
except ImportError:
    minimize_scalar_available = False

__all__ = ["ML_Lag"]


class BaseML_Lag(RegressionPropsY, RegressionPropsVM):

    """
    ML estimation of the spatial lag model (note no consistency
    checks, diagnostics or constants added) :cite:`Anselin1988`

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the Spatial Durbin type.
    method       : string
                   if 'full', brute force calculation (full matrix expressions)
                   if 'ord', Ord eigenvalue method
                   if 'LU', LU sparse matrix decomposition
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and inverse_product

    Attributes
    ----------
    betas        : array
                   (k+1)x1 array of estimated coefficients (rho first)
    rho          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant, excluding the rho)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    method       : string
                   log Jacobian method
                   if 'full': brute force (full matrix computations)
                   if 'ord' : Ord eigenvalue method
    epsilon      : float
                   tolerance criterion used in minimize_scalar function and inverse_product
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+1 x k+1)
    vm1          : array
                   Variance covariance matrix (k+2 x k+2) includes sigma2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    predy_e      : array
                   predicted values from reduced form
    e_pred       : array
                   prediction errors using reduced form predicted values


    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from libpysal.examples import load_example
    >>> import geopandas as gpd
    >>> from libpysal.weights import Queen
    >>> import spreg
    >>> np.set_printoptions(suppress=True) #prevent scientific format
    >>> baltimore = load_example('Baltimore')
    >>> db =  libpysal.io.open(baltimore.get_path("baltim.dbf"),'r')
    >>> df = gpd.read_file(baltimore.get_path("baltim.shp"))
    >>> ds_name = "baltim.dbf"
    >>> y_name = "PRICE"
    >>> y = np.array(db.by_col(y_name)).T
    >>> y.shape = (len(y),1)
    >>> x_names = ["NROOM","NBATH","PATIO","FIREPL","AC","GAR","AGE","LOTSZ","SQFT"]
    >>> x = np.array([db.by_col(var) for var in x_names]).T
    >>> x = np.hstack((np.ones((len(y),1)),x))
    >>> w = Queen.from_dataframe(df)
    >>> w.transform = 'r'
    >>> w_name = "baltim_q.gal"
    >>> mllag = spreg.ml_lag.BaseML_Lag(y,x,w,method='ord') #doctest: +SKIP
    >>> "{0:.6f}".format(mllag.rho) #doctest: +SKIP
    '0.425885'
    >>> np.around(mllag.betas, decimals=4) #doctest: +SKIP
    array([[ 4.3675],
           [ 0.7502],
           [ 5.6116],
           [ 7.0497],
           [ 7.7246],
           [ 6.1231],
           [ 4.6375],
           [-0.1107],
           [ 0.0679],
           [ 0.0794],
           [ 0.4259]])
    >>> "{0:.6f}".format(mllag.mean_y) #doctest: +SKIP
    '44.307180'
    >>> "{0:.6f}".format(mllag.std_y) #doctest: +SKIP
    '23.606077'
    >>> np.around(np.diag(mllag.vm1), decimals=4) #doctest: +SKIP
    array([  23.8716,    1.1222,    3.0593,    7.3416,    5.6695,    5.4698,
              2.8684,    0.0026,    0.0002,    0.0266,    0.0032,  220.1292])
    >>> np.around(np.diag(mllag.vm), decimals=4) #doctest: +SKIP
    array([ 23.8716,   1.1222,   3.0593,   7.3416,   5.6695,   5.4698,
             2.8684,   0.0026,   0.0002,   0.0266,   0.0032])
    >>> "{0:.6f}".format(mllag.sig2) #doctest: +SKIP
    '151.458698'
    >>> "{0:.6f}".format(mllag.logll) #doctest: +SKIP
    '-832.937174'
    >>> mllag = spreg.ml_lag.BaseML_Lag(y,x,w) #doctest: +SKIP
    >>> "{0:.6f}".format(mllag.rho) #doctest: +SKIP
    '0.425885'
    >>> np.around(mllag.betas, decimals=4) #doctest: +SKIP
    array([[ 4.3675],
           [ 0.7502],
           [ 5.6116],
           [ 7.0497],
           [ 7.7246],
           [ 6.1231],
           [ 4.6375],
           [-0.1107],
           [ 0.0679],
           [ 0.0794],
           [ 0.4259]])
    >>> "{0:.6f}".format(mllag.mean_y) #doctest: +SKIP
    '44.307180'
    >>> "{0:.6f}".format(mllag.std_y) #doctest: +SKIP
    '23.606077'
    >>> np.around(np.diag(mllag.vm1), decimals=4) #doctest: +SKIP
    array([  23.8716,    1.1222,    3.0593,    7.3416,    5.6695,    5.4698,
              2.8684,    0.0026,    0.0002,    0.0266,    0.0032,  220.1292])
    >>> np.around(np.diag(mllag.vm), decimals=4) #doctest: +SKIP
    array([ 23.8716,   1.1222,   3.0593,   7.3416,   5.6695,   5.4698,
             2.8684,   0.0026,   0.0002,   0.0266,   0.0032])
    >>> "{0:.6f}".format(mllag.sig2) #doctest: +SKIP
    '151.458698'
    >>> "{0:.6f}".format(mllag.logll) #doctest: +SKIP
    '-832.937174'


    """

    def __init__(self, y, x, w, slx_lags=0, method="full", epsilon=0.0000001):
        # set up main regression variables and spatial filters
        self.y = y
        self.x = x
        self.method = method
        self.epsilon = epsilon
        # W = w.full()[0]
        # Wsp = w.sparse
        ylag = weights.lag_spatial(w, y)
        # b0, b1, e0 and e1

# now set in ML_Lag
#        if slx_lags>0:
#            self.x = np.hstack((self.x, get_lags(w, self.x[:, 1:], slx_lags)))

        self.n, self.k = self.x.shape
        xtx = spdot(self.x.T, self.x)
        xtxi = la.inv(xtx)
        xty = spdot(self.x.T, self.y)
        xtyl = spdot(self.x.T, ylag)
        b0 = spdot(xtxi, xty)
        b1 = spdot(xtxi, xtyl)
        e0 = self.y - spdot(self.x, b0)
        e1 = ylag - spdot(self.x, b1)
        methodML = method.upper()
        # call minimizer using concentrated log-likelihood to get rho
        if methodML in ["FULL", "LU", "ORD"]:
            if methodML == "FULL":
                W = w.full()[0]  # moved here
                res = minimize_scalar(
                    lag_c_loglik,
                    0.0,
                    bounds=(-1.0, 1.0),
                    args=(self.n, e0, e1, W),
                    method="bounded",
                    options={'xatol': epsilon},
                )
            elif methodML == "LU":
                I = sp.identity(w.n)
                Wsp = w.sparse  # moved here
                W = Wsp#.tocsc()
                res = minimize_scalar(
                    lag_c_loglik_sp,
                    0.0,
                    bounds=(-1.0, 1.0),
                    args=(self.n, e0, e1, I, Wsp),
                    method="bounded",
                    options={'xatol': epsilon},
                )
            elif methodML == "ORD":
                # check on symmetry structure
                if w.asymmetry(intrinsic=False) == []:
                    ww = symmetrize(w)
                    WW = np.array(ww.todense())
                    evals = la.eigvalsh(WW)
                    W = WW
                else:
                    W = w.full()[0]  # moved here
                    evals = la.eigvals(W)
                res = minimize_scalar(
                    lag_c_loglik_ord,
                    0.0,
                    bounds=(-1.0, 1.0),
                    args=(self.n, e0, e1, evals),
                    method="bounded",
                    options={'xatol': epsilon},
                )
        else:
            # program will crash, need to catch
            print(("{0} is an unsupported method".format(methodML)))
            self = None
            return

        self.rho = res.x[0][0]

        # compute full log-likelihood, including constants
        ln2pi = np.log(2.0 * np.pi)
        llik = -res.fun - self.n / 2.0 * ln2pi - self.n / 2.0
        self.logll = llik[0][0]

        # b, residuals and predicted values

        b = b0 - self.rho * b1
        self.betas = np.vstack((b, self.rho))  # rho added as last coefficient
        self.u = e0 - self.rho * e1
        self.predy = self.y - self.u

        xb = spdot(self.x, b)

        self.predy_e = inverse_prod(
            w.sparse, xb, self.rho, inv_method="power_exp", threshold=epsilon
        )
        self.e_pred = self.y - self.predy_e

        # residual variance
        self._cache = {}
        self.sig2 = self.sig2n  # no allowance for division by n-k

        # information matrix
        # if w should be kept sparse, how can we do the following:
        a = -self.rho * W
        spfill_diagonal(a, 1.0)
        ai = spinv(a)
        wai = spdot(W, ai)
        tr1 = wai.diagonal().sum()  # same for sparse and dense

        wai2 = spdot(wai, wai)
        tr2 = wai2.diagonal().sum()

        waiTwai = spdot(wai.T, wai)
        tr3 = waiTwai.diagonal().sum()
        ### to here

        wpredy = weights.lag_spatial(w, self.predy_e)
        wpyTwpy = spdot(wpredy.T, wpredy)
        xTwpy = spdot(self.x.T, wpredy)

        # order of variables is beta, rho, sigma2

        v1 = np.vstack((xtx / self.sig2, xTwpy.T / self.sig2, np.zeros((1, self.k))))
        v2 = np.vstack(
            (xTwpy / self.sig2, tr2 + tr3 + wpyTwpy / self.sig2, tr1 / self.sig2)
        )
        v3 = np.vstack(
            (np.zeros((self.k, 1)), tr1 / self.sig2, self.n / (2.0 * self.sig2 ** 2))
        )

        v = np.hstack((v1, v2, v3))

        self.vm1 = la.inv(v)  # vm1 includes variance for sigma2
        self.vm = self.vm1[:-1, :-1]  # vm is for coefficients only


class ML_Lag(BaseML_Lag):

    """
    ML estimation of the spatial lag model with all results and diagnostics; :cite:`Anselin1988`

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the Spatial Durbin type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
    regimes      : list or pandas.Series
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
                   For other regimes-specific arguments, see ML_Lag_Regimes
    method       : string
                   if 'full', brute force calculation (full matrix expressions)
                   if 'ord', Ord eigenvalue method
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and inverse_product
    spat_diag    : boolean
                   If True, then compute Common Factor Hypothesis test when applicable
    spat_impacts : string or list
                   Include average direct impact (ADI), average indirect impact (AII),
                    and average total impact (ATI) in summary results.
                    Options are 'simple', 'full', 'power', 'all' or None.
                    See sputils.spmultiplier for more information.
    vm           : boolean
                   if True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    latex        : boolean
                   Specifies if summary is to be printed in latex format

    Attributes
    ----------
    output       : dataframe
                   regression results pandas dataframe
    betas        : array
                   (k+1)x1 array of estimated coefficients (rho first)
    rho          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant, excluding the rho)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    method       : string
                   log Jacobian method
                   if 'full': brute force (full matrix computations)
    epsilon      : float
                   tolerance criterion used in minimize_scalar function and inverse_product
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+1 x k+1), all coefficients
    vm1          : array
                   Variance covariance matrix (k+2 x k+2), includes sig2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    aic          : float
                   Akaike information criterion
    schwarz      : float
                   Schwarz criterion
    cfh_test     : tuple
                   Common Factor Hypothesis test; tuple contains the pair (statistic,
                   p-value). Only when it applies (see specific documentation).
    predy_e      : array
                   predicted values from reduced form
    e_pred       : array
                   prediction errors using reduced form predicted values
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
    utu          : float
                   Sum of squared residuals
    std_err      : array
                   1xk array of standard errors of the betas
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    sp_multipliers: dict
                   Dictionary of spatial multipliers (if spat_impacts is not None)             
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    title        : string
                   Name of the regression method used

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from libpysal.examples import load_example
    >>> from libpysal.weights import Queen
    >>> from spreg import ML_Error_Regimes
    >>> import geopandas as gpd
    >>> from spreg import ML_Lag
    >>> np.set_printoptions(suppress=True) #prevent scientific format
    >>> baltimore = load_example('Baltimore')
    >>> db = libpysal.io.open(baltimore.get_path("baltim.dbf"),'r')
    >>> df = gpd.read_file(baltimore.get_path("baltim.shp"))
    >>> ds_name = "baltim.dbf"
    >>> y_name = "PRICE"
    >>> y = np.array(db.by_col(y_name)).T
    >>> y.shape = (len(y),1)
    >>> x_names = ["NROOM","NBATH","PATIO","FIREPL","AC","GAR","AGE","LOTSZ","SQFT"]
    >>> x = np.array([db.by_col(var) for var in x_names]).T
    >>> w = Queen.from_dataframe(df)
    >>> w_name = "baltim_q.gal"
    >>> w.transform = 'r'
    >>> mllag = ML_Lag(y,x,w,name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name) #doctest: +SKIP
    >>> np.around(mllag.betas, decimals=4) #doctest: +SKIP
    array([[ 4.3675],
           [ 0.7502],
           [ 5.6116],
           [ 7.0497],
           [ 7.7246],
           [ 6.1231],
           [ 4.6375],
           [-0.1107],
           [ 0.0679],
           [ 0.0794],
           [ 0.4259]])
    >>> "{0:.6f}".format(mllag.rho) #doctest: +SKIP
    '0.425885'
    >>> "{0:.6f}".format(mllag.mean_y) #doctest: +SKIP
    '44.307180'
    >>> "{0:.6f}".format(mllag.std_y) #doctest: +SKIP
    '23.606077'
    >>> np.around(np.diag(mllag.vm1), decimals=4) #doctest: +SKIP
    array([  23.8716,    1.1222,    3.0593,    7.3416,    5.6695,    5.4698,
              2.8684,    0.0026,    0.0002,    0.0266,    0.0032,  220.1292])
    >>> np.around(np.diag(mllag.vm), decimals=4) #doctest: +SKIP
    array([ 23.8716,   1.1222,   3.0593,   7.3416,   5.6695,   5.4698,
             2.8684,   0.0026,   0.0002,   0.0266,   0.0032])
    >>> "{0:.6f}".format(mllag.sig2) #doctest: +SKIP
    '151.458698'
    >>> "{0:.6f}".format(mllag.logll) #doctest: +SKIP
    '-832.937174'
    >>> "{0:.6f}".format(mllag.aic) #doctest: +SKIP
    '1687.874348'
    >>> "{0:.6f}".format(mllag.schwarz) #doctest: +SKIP
    '1724.744787'
    >>> "{0:.6f}".format(mllag.pr2) #doctest: +SKIP
    '0.727081'
    >>> "{0:.4f}".format(mllag.pr2_e) #doctest: +SKIP
    '0.7062'
    >>> "{0:.4f}".format(mllag.utu) #doctest: +SKIP
    '31957.7853'
    >>> np.around(mllag.std_err, decimals=4) #doctest: +SKIP
    array([ 4.8859,  1.0593,  1.7491,  2.7095,  2.3811,  2.3388,  1.6936,
            0.0508,  0.0146,  0.1631,  0.057 ])
    >>> np.around(mllag.z_stat, decimals=4) #doctest: +SKIP
    array([[ 0.8939,  0.3714],
           [ 0.7082,  0.4788],
           [ 3.2083,  0.0013],
           [ 2.6018,  0.0093],
           [ 3.2442,  0.0012],
           [ 2.6181,  0.0088],
           [ 2.7382,  0.0062],
           [-2.178 ,  0.0294],
           [ 4.6487,  0.    ],
           [ 0.4866,  0.6266],
           [ 7.4775,  0.    ]])
    >>> mllag.name_y #doctest: +SKIP
    'PRICE'
    >>> mllag.name_x #doctest: +SKIP
    ['CONSTANT', 'NROOM', 'NBATH', 'PATIO', 'FIREPL', 'AC', 'GAR', 'AGE', 'LOTSZ', 'SQFT', 'W_PRICE']
    >>> mllag.name_w #doctest: +SKIP
    'baltim_q.gal'
    >>> mllag.name_ds #doctest: +SKIP
    'baltim.dbf'
    >>> mllag.title #doctest: +SKIP
    'MAXIMUM LIKELIHOOD SPATIAL LAG (METHOD = FULL)'
    >>> mllag = ML_Lag(y,x,w,method='ord',name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name) #doctest: +SKIP
    >>> np.around(mllag.betas, decimals=4) #doctest: +SKIP
    array([[ 4.3675],
           [ 0.7502],
           [ 5.6116],
           [ 7.0497],
           [ 7.7246],
           [ 6.1231],
           [ 4.6375],
           [-0.1107],
           [ 0.0679],
           [ 0.0794],
           [ 0.4259]])
    >>> "{0:.6f}".format(mllag.rho) #doctest: +SKIP
    '0.425885'
    >>> "{0:.6f}".format(mllag.mean_y) #doctest: +SKIP
    '44.307180'
    >>> "{0:.6f}".format(mllag.std_y) #doctest: +SKIP
    '23.606077'
    >>> np.around(np.diag(mllag.vm1), decimals=4) #doctest: +SKIP
    array([  23.8716,    1.1222,    3.0593,    7.3416,    5.6695,    5.4698,
              2.8684,    0.0026,    0.0002,    0.0266,    0.0032,  220.1292])
    >>> np.around(np.diag(mllag.vm), decimals=4) #doctest: +SKIP
    array([ 23.8716,   1.1222,   3.0593,   7.3416,   5.6695,   5.4698,
             2.8684,   0.0026,   0.0002,   0.0266,   0.0032])
    >>> "{0:.6f}".format(mllag.sig2) #doctest: +SKIP
    '151.458698'
    >>> "{0:.6f}".format(mllag.logll) #doctest: +SKIP
    '-832.937174'
    >>> "{0:.6f}".format(mllag.aic) #doctest: +SKIP
    '1687.874348'
    >>> "{0:.6f}".format(mllag.schwarz) #doctest: +SKIP
    '1724.744787'
    >>> "{0:.6f}".format(mllag.pr2) #doctest: +SKIP
    '0.727081'
    >>> "{0:.6f}".format(mllag.pr2_e) #doctest: +SKIP
    '0.706198'
    >>> "{0:.4f}".format(mllag.utu) #doctest: +SKIP
    '31957.7853'
    >>> np.around(mllag.std_err, decimals=4) #doctest: +SKIP
    array([ 4.8859,  1.0593,  1.7491,  2.7095,  2.3811,  2.3388,  1.6936,
            0.0508,  0.0146,  0.1631,  0.057 ])
    >>> np.around(mllag.z_stat, decimals=4) #doctest: +SKIP
    array([[ 0.8939,  0.3714],
           [ 0.7082,  0.4788],
           [ 3.2083,  0.0013],
           [ 2.6018,  0.0093],
           [ 3.2442,  0.0012],
           [ 2.6181,  0.0088],
           [ 2.7382,  0.0062],
           [-2.178 ,  0.0294],
           [ 4.6487,  0.    ],
           [ 0.4866,  0.6266],
           [ 7.4775,  0.    ]])
    >>> mllag.name_y #doctest: +SKIP
    'PRICE'
    >>> mllag.name_x #doctest: +SKIP
    ['CONSTANT', 'NROOM', 'NBATH', 'PATIO', 'FIREPL', 'AC', 'GAR', 'AGE', 'LOTSZ', 'SQFT', 'W_PRICE']
    >>> mllag.name_w #doctest: +SKIP
    'baltim_q.gal'
    >>> mllag.name_ds #doctest: +SKIP
    'baltim.dbf'
    >>> mllag.title #doctest: +SKIP
    'MAXIMUM LIKELIHOOD SPATIAL LAG (METHOD = ORD)'


    """

    def __init__(
        self,
        y,
        x,
        w,
        slx_lags=0,
        slx_vars="All",
        regimes=None,
        method="full",
        epsilon=0.0000001,
        spat_impacts="simple",
        vm=False,
        spat_diag=True,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
        **kwargs
    ):
        if regimes is not None:
            from .ml_lag_regimes import ML_Lag_Regimes
            self.__class__ = ML_Lag_Regimes
            self.__init__(
                y=y,
                x=x,
                regimes=regimes,
                w=w,
                slx_lags=slx_lags,
                method=method,
                epsilon=epsilon,
                spat_impacts=spat_impacts,
                vm=vm,
                spat_diag=spat_diag,
                name_y=name_y,
                name_x=name_x,
                name_w=name_w,
                name_ds=name_ds,
                latex=latex,
                **kwargs
            )
        else:
            n = USER.check_arrays(y, x)
            y, name_y = USER.check_y(y, n, name_y)
            w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
            x_constant, name_x, warn = USER.check_constant(x, name_x)
            name_x = USER.set_name_x(name_x, x_constant) # needs to be initialized for none, now with constant
            set_warn(self, warn)
            method = method.upper()
            # using flex_wx
            kx = len(name_x)
            if slx_lags > 0:
                x_constant,name_x = USER.flex_wx(w,x=x_constant,name_x=name_x,constant=True,
                                                slx_lags=slx_lags,slx_vars=slx_vars)
                if isinstance(slx_vars,list):
                    kw = slx_vars.count(True)
                    if kw < kx - 1:
                        spat_diag = False   # no common factor test
                else:
                    kw = kx-1
        

            BaseML_Lag.__init__(
                self, y=y, x=x_constant, w=w, slx_lags=slx_lags, method=method, epsilon=epsilon
            )
            # increase by 1 to have correct aic and sc, include rho in count
            self.k += 1

            if slx_lags>0:
        #        kx = len(name_x)
        #        name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)  # exclude constant

                self.title = "MAXIMUM LIKELIHOOD SPATIAL LAG WITH SLX - SPATIAL DURBIN MODEL" + " (METHOD = " + method + ")"
                var_types = ['o'] + ['x'] * (kx-1) + ['wx'] * (kw) * slx_lags + ['rho']
            else:
                self.title = "MAXIMUM LIKELIHOOD SPATIAL LAG" + " (METHOD = " + method + ")"
                var_types = ['o'] + ['x'] * (len(name_x)-1) + ['rho']
            self.slx_lags = slx_lags
            self.slx_vars = slx_vars
            self.name_ds = USER.set_name_ds(name_ds)
            self.name_y = USER.set_name_y(name_y)
            self.name_x = name_x  # already has constant
            name_ylag = USER.set_name_yend_sp(self.name_y)
            self.name_x.append(name_ylag)  # rho changed to last position
            self.name_w = USER.set_name_w(name_w, w)
            self.aic = DIAG.akaike(reg=self)
            self.schwarz = DIAG.schwarz(reg=self)
            self.output = pd.DataFrame(self.name_x, columns=['var_names'])
            self.output['var_type'] = var_types
            self.output['regime'], self.output['equation'] = (0, 0)
            self.other_top = _spat_pseudo_r2(self)
            self.other_top += _nonspat_top(self, ml=True)
            diag_out = None
            if spat_diag and slx_lags==1:
                diag_out = _spat_diag_out(self, w, 'yend', ml=True)
            if spat_impacts:
                self.sp_multipliers, impacts_str = _summary_impacts(self, w, spat_impacts, slx_lags,slx_vars)
                try:
                    diag_out += impacts_str
                except TypeError:
                    diag_out = impacts_str
            output(reg=self, vm=vm, robust=False, other_end=diag_out, latex=latex)

def lag_c_loglik(rho, n, e0, e1, W):
    # concentrated log-lik for lag model, no constants, brute force
    er = e0 - rho * e1
    sig2 = spdot(er.T, er) / n
    nlsig2 = (n / 2.0) * np.log(sig2)
    a = -rho * W
    spfill_diagonal(a, 1.0)
    jacob = np.log(np.linalg.det(a))
    # this is the negative of the concentrated log lik for minimization
    clik = nlsig2 - jacob
    return clik


def lag_c_loglik_sp(rho, n, e0, e1, I, Wsp):
    # concentrated log-lik for lag model, sparse algebra
    if isinstance(rho, np.ndarray):
        if rho.shape == (1, 1):
            rho = rho[0][0]  # why does the interior value change?
    er = e0 - rho * e1
    sig2 = spdot(er.T, er) / n
    nlsig2 = (n / 2.0) * np.log(sig2)
    a = I - rho * Wsp
    LU = SuperLU(a.tocsc())
    jacob = np.sum(np.log(np.abs(LU.U.diagonal())))
    clike = nlsig2 - jacob
    return clike


def lag_c_loglik_ord(rho, n, e0, e1, evals):
    # concentrated log-lik for lag model, no constants, Ord eigenvalue method
    er = e0 - rho * e1
    sig2 = spdot(er.T, er) / n
    nlsig2 = (n / 2.0) * np.log(sig2)
    revals = rho * evals
    jacob = np.log(1 - revals).sum()
    if isinstance(jacob, complex):
        jacob = jacob.real
    # this is the negative of the concentrated log lik for minimization
    clik = nlsig2 - jacob
    return clik


def _test():
    import doctest

    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)

if __name__ == "__main__":
    _test()

    import numpy as np
    import libpysal

    db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    y_var = "CRIME"
    y = np.array([db.by_col(y_var)]).reshape(49, 1)
    x_var = ["INC"]
    x = np.array([db.by_col(name) for name in x_var]).T
    w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    w.transform = "r"
    model = ML_Lag(
        y,
        x,
        w=w,
        vm=False,
        name_y=y_var,
        name_x=x_var,
        name_ds="columbus",
        name_w="columbus.gal",
    )
    print(model.output)
    print(model.summary)
