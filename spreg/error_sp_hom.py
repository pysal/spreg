"""
Hom family of models based on: :cite:`Drukker2013`
Following: :cite:`Anselin2011`

"""

__author__ = "Luc Anselin lanselin@gmail.com, Daniel Arribas-Bel darribas@asu.edu"

from scipy import sparse as SP
import numpy as np
from numpy import linalg as la
from . import ols as OLS
from .utils import set_endog, iter_msg, sp_att
from .utils import get_A1_hom, get_A2_hom, get_A1_het, optim_moments
from .utils import get_spFilter, get_lags
from .utils import spdot, RegressionPropsY, set_warn
from . import twosls as TSLS
from . import user_output as USER
import pandas as pd
from .output import output, _spat_pseudo_r2, _summary_iteration
from itertools import compress

__all__ = ["GM_Error_Hom", "GM_Endog_Error_Hom", "GM_Combo_Hom"]


class BaseGM_Error_Hom(RegressionPropsY):
    """
    GMM method for a spatial error model with homoskedasticity (note: no
    consistency checks, diagnostics or constant added); based on
    Drukker et al. (2013) :cite:`Drukker2013`, following Anselin (2011) :cite:`Anselin2011`.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : Sparse matrix
                   Spatial weights sparse matrix
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from :cite:`Arraiz2010`.
                   Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from :cite:`Arraiz2010`. Note: max_iter provides
                   an additional stop condition.
    A1           : string
                   If A1='het', then the matrix A1 is defined as in :cite:`Arraiz2010`.
                   If A1='hom', then as in :cite:`Anselin2011` (default).  If
                   A1='hom_sc' (default), then as in :cite:`Drukker2013`
                   and :cite:`Drukker:2013aa`.
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from :cite:`Arraiz2010`.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from :cite:`Arraiz2010`.
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    xtx          : float
                   :math:`X'X`

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = 'r'

    Model commands

    >>> reg = BaseGM_Error_Hom(y, X, w=w.sparse, A1='hom_sc')
    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[47.9479 12.3021]
     [ 0.7063  0.4967]
     [-0.556   0.179 ]
     [ 0.4129  0.1835]]
    >>> print(np.around(reg.vm, 4)) #doctest: +SKIP
    [[  1.51340700e+02  -5.29060000e+00  -1.85650000e+00  -2.40000000e-03]
     [ -5.29060000e+00   2.46700000e-01   5.14000000e-02   3.00000000e-04]
     [ -1.85650000e+00   5.14000000e-02   3.21000000e-02  -1.00000000e-04]
     [ -2.40000000e-03   3.00000000e-04  -1.00000000e-04   3.37000000e-02]]
    """

    def __init__(
        self, y, x, w, max_iter=1, epsilon=0.00001, A1="hom_sc", hard_bound=False
    ):
        if A1 == "hom":
            wA1 = get_A1_hom(w)
        elif A1 == "hom_sc":
            wA1 = get_A1_hom(w, scalarKP=True)
        elif A1 == "het":
            wA1 = get_A1_het(w)

        wA2 = get_A2_hom(w)

        # 1a. OLS --> \tilde{\delta}
        ols = OLS.BaseOLS(y=y, x=x)
        self.x, self.y, self.n, self.k, self.xtx = ols.x, ols.y, ols.n, ols.k, ols.xtx

        # 1b. GM --> \tilde{\rho}
        moments = moments_hom(w, wA1, wA2, ols.u)
        lambda1 = optim_moments(moments, hard_bound=hard_bound)
        lambda_old = lambda1

        self.iteration, eps = 0, 1
        while self.iteration < max_iter and eps > epsilon:
            # 2a. SWLS --> \hat{\delta}
            x_s = get_spFilter(w, lambda_old, self.x)
            y_s = get_spFilter(w, lambda_old, self.y)
            ols_s = OLS.BaseOLS(y=y_s, x=x_s)
            self.predy = spdot(self.x, ols_s.betas)
            self.u = self.y - self.predy

            # 2b. GM 2nd iteration --> \hat{\rho}
            moments = moments_hom(w, wA1, wA2, self.u)
            psi = get_vc_hom(w, wA1, wA2, self, lambda_old)[0]
            lambda2 = optim_moments(moments, psi, hard_bound=hard_bound)
            eps = abs(lambda2 - lambda_old)
            lambda_old = lambda2
            self.iteration += 1

        self.iter_stop = iter_msg(self.iteration, max_iter)

        # Output
        self.betas = np.vstack((ols_s.betas, lambda2))
        self.vm, self.sig2 = get_omega_hom_ols(w, wA1, wA2, self, lambda2, moments[0])
        self.e_filtered = self.u - lambda2 * w * self.u
        self._cache = {}


class GM_Error_Hom(BaseGM_Error_Hom):
    """
    GMM method for a spatial error model with homoskedasticity, with results
    and diagnostics; based on Drukker et al. (2013) :cite:`Drukker2013`, following Anselin
    (2011) :cite:`Anselin2011`.

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
                   If slx_lags>0, the specification becomes of the SLX-Error type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from :cite:`Arraiz2010`.
                   Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from :cite:`Arraiz2010`. Note: max_iter provides
                   an additional stop condition.
    A1           : string
                   If A1='het', then the matrix A1 is defined as in Arraiz et
                   al. If A1='hom', then as in :cite:`Anselin2011`.  If
                   A1='hom_sc' (default), then as in :cite:`Drukker2013`
                   and :cite:`Drukker:2013aa`.
    vm           : boolean
                   If True, include variance-covariance matrix in summary
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
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    Attributes
    ----------
    output       : dataframe
                   regression results pandas dataframe
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from :cite:`Arraiz2010`.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from Arraiz et al.
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    xtx          : float
                   :math:`X'X`
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

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import libpysal

    Open data on Columbus neighborhood crime (49 areas) using libpysal.io.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')

    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) and CRIME (crime) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
    independent variables passed in.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``columbus.shp``.

    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> reg = GM_Error_Hom(y, X, w=w, A1='hom_sc', name_y='home value', name_x=['income', 'crime'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. This class offers an error model that assumes
    homoskedasticity but that unlike the models from
    ``spreg.error_sp``, it allows for inference on the spatial
    parameter. This is why you obtain as many coefficient estimates as
    standard errors, which you calculate taking the square root of the
    diagonal of the variance-covariance matrix of the parameters:

    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[47.9479 12.3021]
     [ 0.7063  0.4967]
     [-0.556   0.179 ]
     [ 0.4129  0.1835]]

    """

    def __init__(
        self,
        y,
        x,
        w,
        slx_lags=0,
        slx_vars="All",
        max_iter=1,
        epsilon=0.00001,
        A1="hom_sc",
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
        hard_bound=False,
    ):
        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        x_constant, name_x, warn = USER.check_constant(x, name_x)
        name_x = USER.set_name_x(
            name_x, x_constant
        )  # initialize in case None, includes constant
        set_warn(self, warn)
        self.title = "GM SPATIALLY WEIGHTED LEAST SQUARES (HOM)"

        if slx_lags > 0:
            x_constant, name_x = USER.flex_wx(
                w,
                x=x_constant,
                name_x=name_x,
                constant=True,
                slx_lags=slx_lags,
                slx_vars=slx_vars,
            )
            self.title += " WITH SLX (SLX-Error)"

        # OLD
        # if slx_lags >0:
        # lag_x = get_lags(w, x_constant[:, 1:], slx_lags)
        # x_constant = np.hstack((x_constant, lag_x))
        #            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
        # name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)  # exclude constant
        # self.title += " WITH SLX (SLX-Error)"

        BaseGM_Error_Hom.__init__(
            self,
            y=y,
            x=x_constant,
            w=w.sparse,
            A1=A1,
            max_iter=max_iter,
            epsilon=epsilon,
            hard_bound=hard_bound,
        )
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        #        self.name_x = USER.set_name_x(name_x, x_constant)
        self.name_x = name_x  # constant already included
        self.name_x.append("lambda")
        self.name_w = USER.set_name_w(name_w, w)
        self.A1 = A1
        self.output = pd.DataFrame(self.name_x, columns=["var_names"])
        self.output["var_type"] = ["x"] * (len(self.name_x) - 1) + ["lambda"]
        self.output["regime"], self.output["equation"] = (0, 0)
        self.other_top = _summary_iteration(self)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


class BaseGM_Endog_Error_Hom(RegressionPropsY):
    """
    GMM method for a spatial error model with homoskedasticity and
    endogenous variables (note: no consistency checks, diagnostics or constant
    added); based on Drukker et al. (2013) :cite:`Drukker2013`, following Anselin (2011)
    :cite:`Anselin2011`.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note:
                   this should not contain any variables from x)
    w            : Sparse matrix
                   Spatial weights sparse matrix
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from
                   :cite:`Arraiz2010`. Note: epsilon provides an additional
                   stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from :cite:`Arraiz2010`. Note: max_iter provides
                   an additional stop condition.
    A1           : string
                   If A1='het', then the matrix A1 is defined as in Arraiz et
                   al. If A1='hom', then as in :cite:`Anselin2011`.  If
                   A1='hom_sc' (default), then as in :cite:`Drukker2013`
                   and :cite:`Drukker:2013aa`.
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from :cite:`Arraiz2010`.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from :cite:`Arraiz2010`.
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    hth          : float
                   H'H

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = 'r'
    >>> reg = BaseGM_Endog_Error_Hom(y, X, yd, q, w=w.sparse, A1='hom_sc')
    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[55.3658 23.496 ]
     [ 0.4643  0.7382]
     [-0.669   0.3943]
     [ 0.4321  0.1927]]


    """

    def __init__(
        self,
        y,
        x,
        yend,
        q,
        w,
        max_iter=1,
        epsilon=0.00001,
        A1="hom_sc",
        hard_bound=False,
    ):
        if A1 == "hom":
            wA1 = get_A1_hom(w)
        elif A1 == "hom_sc":
            wA1 = get_A1_hom(w, scalarKP=True)
        elif A1 == "het":
            wA1 = get_A1_het(w)

        wA2 = get_A2_hom(w)

        # 1a. S2SLS --> \tilde{\delta}
        tsls = TSLS.BaseTSLS(y=y, x=x, yend=yend, q=q)
        self.x, self.z, self.h, self.y, self.hth = (
            tsls.x,
            tsls.z,
            tsls.h,
            tsls.y,
            tsls.hth,
        )
        self.yend, self.q, self.n, self.k = tsls.yend, tsls.q, tsls.n, tsls.k

        # 1b. GM --> \tilde{\rho}
        moments = moments_hom(w, wA1, wA2, tsls.u)
        lambda1 = optim_moments(moments, hard_bound=hard_bound)
        lambda_old = lambda1

        self.iteration, eps = 0, 1
        while self.iteration < max_iter and eps > epsilon:
            # 2a. GS2SLS --> \hat{\delta}
            x_s = get_spFilter(w, lambda_old, self.x)
            y_s = get_spFilter(w, lambda_old, self.y)
            yend_s = get_spFilter(w, lambda_old, self.yend)
            tsls_s = TSLS.BaseTSLS(y=y_s, x=x_s, yend=yend_s, h=self.h)
            self.predy = spdot(self.z, tsls_s.betas)
            self.u = self.y - self.predy

            # 2b. GM 2nd iteration --> \hat{\rho}
            moments = moments_hom(w, wA1, wA2, self.u)
            psi = get_vc_hom(w, wA1, wA2, self, lambda_old, tsls_s.z)[0]
            lambda2 = optim_moments(moments, psi, hard_bound=hard_bound)
            eps = abs(lambda2 - lambda_old)
            lambda_old = lambda2
            self.iteration += 1

        self.iter_stop = iter_msg(self.iteration, max_iter)

        # Output
        self.betas = np.vstack((tsls_s.betas, lambda2))
        self.vm, self.sig2 = get_omega_hom(w, wA1, wA2, self, lambda2, moments[0])
        self.e_filtered = self.u - lambda2 * w * self.u
        self._cache = {}


class GM_Endog_Error_Hom(BaseGM_Endog_Error_Hom):
    """
    GMM method for a spatial error model with homoskedasticity and endogenous
    variables, with results and diagnostics; based on Drukker et al. (2013)
    :cite:`Drukker2013`, following Anselin (2011) :cite:`Anselin2011`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note:
                   this should not contain any variables from x)
    w            : pysal W object
                   Spatial weights object
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX-Error type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from
                   :cite:`Arraiz2010`. Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from :cite:`Arraiz2010`. Note: max_iter provides
                   an additional stop condition.
    A1           : string
                   If A1='het', then the matrix A1 is defined as in :cite:`Arraiz2010`.
                   If A1='hom', then as in :cite:`Anselin2011`. If
                   A1='hom_sc' (default), then as in :cite:`Drukker2013`
                   and :cite:`Drukker:2013aa`.
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    Attributes
    ----------
    output       : dataframe
                   regression results pandas dataframe
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from :cite:`Arraiz2010`.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from :cite:`Arraiz2010`.
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    sig2         : float
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_yend     : list of strings
                    Names of endogenous variables for use in output
    name_z        : list of strings
                    Names of exogenous and endogenous variables for use in
                    output
    name_q        : list of strings
                    Names of external instruments
    name_h        : list of strings
                    Names of all instruments used in ouput
    name_w        : string
                    Name of weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    title         : string
                    Name of the regression method used
    hth          : float
                   :math:`H'H`


    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import libpysal

    Open data on Columbus neighborhood crime (49 areas) using libpysal.io.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')

    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
    independent variables passed in.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T

    In this case we consider CRIME (crime rates) is an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for CRIME. We use DISCBD (distance to the
    CBD) for this and hence put it in the instruments parameter, 'q'.

    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``columbus.shp``.

    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables (exogenous and endogenous), the
    instruments and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> reg = GM_Endog_Error_Hom(y, X, yd, q, w=w, A1='hom_sc', name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. This class offers an error model that assumes
    homoskedasticity but that unlike the models from
    ``spreg.error_sp``, it allows for inference on the spatial
    parameter. Hence, we find the same number of betas as of standard errors,
    which we calculate taking the square root of the diagonal of the
    variance-covariance matrix:

    >>> print(reg.name_z)
    ['CONSTANT', 'inc', 'crime', 'lambda']
    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[55.3658 23.496 ]
     [ 0.4643  0.7382]
     [-0.669   0.3943]
     [ 0.4321  0.1927]]
    """

    def __init__(
        self,
        y,
        x,
        yend,
        q,
        w,
        slx_lags=0,
        slx_vars="All",
        max_iter=1,
        epsilon=0.00001,
        A1="hom_sc",
        vm=False,
        name_y=None,
        name_x=None,
        name_yend=None,
        name_q=None,
        name_w=None,
        name_ds=None,
        latex=False,
        hard_bound=False,
    ):
        n = USER.check_arrays(y, x, yend, q)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        yend, q, name_yend, name_q = USER.check_endog([yend, q], [name_yend, name_q])
        x_constant, name_x, warn = USER.check_constant(x, name_x)
        name_x = USER.set_name_x(
            name_x, x_constant
        )  # initialize in case None, includes constant
        set_warn(self, warn)
        self.title = "GM SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES (HOM)"

        if slx_lags > 0:
            x_constant, name_x = USER.flex_wx(
                w,
                x=x_constant,
                name_x=name_x,
                constant=True,
                slx_lags=slx_lags,
                slx_vars=slx_vars,
            )

            self.title += " WITH SLX (SLX-Error)"

        # OLD
        # if slx_lags > 0:
        # lag_x = get_lags(w, x_constant[:, 1:], slx_lags)
        # x_constant = np.hstack((x_constant, lag_x))
        #            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
        # name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)  # exclude constant
        # self.title += " WITH SLX (SLX-Error)"

        BaseGM_Endog_Error_Hom.__init__(
            self,
            y=y,
            x=x_constant,
            w=w.sparse,
            yend=yend,
            q=q,
            A1=A1,
            max_iter=max_iter,
            epsilon=epsilon,
            hard_bound=hard_bound,
        )
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        #        self.name_x = USER.set_name_x(name_x, x_constant)
        self.name_x = name_x  # already includes constant
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append("lambda")  # listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        self.A1 = A1
        self.output = pd.DataFrame(self.name_z, columns=["var_names"])
        self.output["var_type"] = (
            ["x"] * len(self.name_x) + ["yend"] * len(self.name_yend) + ["lambda"]
        )
        self.output["regime"], self.output["equation"] = (0, 0)
        self.other_top = _summary_iteration(self)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


class BaseGM_Combo_Hom(BaseGM_Endog_Error_Hom):
    """
    GMM method for a spatial lag and error model with homoskedasticity and
    endogenous variables (note: no consistency checks, diagnostics or constant
    added); based on Drukker et al. (2013) :cite:`Drukker2013`, following Anselin (2011)
    :cite:`Anselin2011`.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note:
                   this should not contain any variables from x)
    w            : Sparse matrix
                   Spatial weights sparse matrix
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional
                   instruments (q).
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from :cite:`Arraiz2010`.
                   Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from :cite:`Arraiz2010`. Note: max_iter provides
                   an additional stop condition.
    A1           : string
                   If A1='het', then the matrix A1 is defined as in Arraiz et
                   al. If A1='hom', then as in :cite:`Anselin2011`.  If
                   A1='hom_sc' (default), then as in :cite:`Drukker2013`
                   and :cite:`Drukker:2013aa`.
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.

    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from :cite:`Arraiz2010`.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from :cite:`Arraiz2010`.
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    hth          : float
                   :math:`H'H`


    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> import spreg
    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = 'r'
    >>> w_lags = 1
    >>> yd2, q2 = spreg.set_endog(y, X, w, None, None, w_lags, True)
    >>> X = np.hstack((np.ones(y.shape),X))

    Example only with spatial lag

    >>> reg = spreg.error_sp_hom.BaseGM_Combo_Hom(y, X, yend=yd2, q=q2, w=w.sparse, A1='hom_sc')
    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[10.1254 15.2871]
     [ 1.5683  0.4407]
     [ 0.1513  0.4048]
     [ 0.2103  0.4226]]


    Example with both spatial lag and other endogenous variables

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> yd2, q2 = spreg.set_endog(y, X, w, yd, q, w_lags, True)
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> reg = spreg.error_sp_hom.BaseGM_Combo_Hom(y, X, yd2, q2, w=w.sparse, A1='hom_sc')
    >>> betas = np.array([['CONSTANT'],['inc'],['crime'],['W_hoval'],['lambda']])
    >>> print(np.hstack((betas, np.around(np.hstack((reg.betas, np.sqrt(reg.vm.diagonal()).reshape(5,1))),5))))
    [['CONSTANT' '111.77057' '67.75191']
     ['inc' '-0.30974' '1.16656']
     ['crime' '-1.36043' '0.6841']
     ['W_hoval' '-0.52908' '0.84428']
     ['lambda' '0.60116' '0.18605']]

    """

    def __init__(
        self,
        y,
        x,
        yend=None,
        q=None,
        w=None,
        w_lags=1,
        lag_q=True,
        max_iter=1,
        epsilon=0.00001,
        A1="hom_sc",
        hard_bound=False,
    ):
        BaseGM_Endog_Error_Hom.__init__(
            self,
            y=y,
            x=x,
            w=w,
            yend=yend,
            q=q,
            A1=A1,
            max_iter=max_iter,
            epsilon=epsilon,
            hard_bound=hard_bound,
        )


class GM_Combo_Hom(BaseGM_Combo_Hom):
    """
    GMM method for a spatial lag and error model with homoskedasticity and
    endogenous variables, with results and diagnostics; based on Drukker et
    al. (2013) :cite:`Drukker2013`, following Anselin (2011) :cite:`Anselin2011`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note: 
                   this should not contain any variables from x)
    w            : pysal W object
                   Spatial weights object (always necessary)   
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the General Nesting
                   Spatial Model (GNSM) type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
    lag_q        : boolean
                   If True, then include spatial lags of the additional 
                   instruments (q).
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from
                   :cite:`Arraiz2010`. Note: epsilon provides an additional
                   stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from :cite:`Arraiz2010`. Note: max_iter provides
                   an additional stop condition.
    A1           : string
                   If A1='het', then the matrix A1 is defined as in :cite:`Arraiz2010`.
                   If A1='hom', then as in :cite:`Anselin2011`.  If
                   A1='hom_sc' (default), then as in :cite:`Drukker2013`
                   and :cite:`Drukker:2013aa`.
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    Attributes
    ----------
    output       : dataframe
                   regression results pandas dataframe
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    e_pred       : array
                   nx1 array of residuals (using reduced form)
    predy        : array
                   nx1 array of predicted y values
    predy_e      : array
                   nx1 array of predicted y values (using reduced form)
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments 
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from :cite:`Arraiz2010`.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from :cite:`Arraiz2010`.
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
    sig2         : float
                   Sigma squared used in computations (based on filtered
                   residuals)
    std_err      : array
                   1xk array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_yend     : list of strings
                    Names of endogenous variables for use in output
    name_z        : list of strings
                    Names of exogenous and endogenous variables for use in 
                    output
    name_q        : list of strings
                    Names of external instruments
    name_h        : list of strings
                    Names of all instruments used in ouput
    name_w        : string
                    Name of weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    title         : string
                    Name of the regression method used
    hth          : float
                   :math:`H'H`


    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import libpysal

    Open data on Columbus neighborhood crime (49 areas) using libpysal.io.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')

    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
    independent variables passed in.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``columbus.shp``.

    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    Example only with spatial lag

    The Combo class runs an SARAR model, that is a spatial lag+error model.
    In this case we will run a simple version of that, where we have the
    spatial effects as well as exogenous variables. Since it is a spatial
    model, we have to pass in the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> from spreg import GM_Combo_Hom
    >>> reg = GM_Combo_Hom(y, X, w=w, A1='hom_sc', name_x=['inc'],\
            name_y='hoval', name_yend=['crime'], name_q=['discbd'],\
            name_ds='columbus')
    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[10.1254 15.2871]
     [ 1.5683  0.4407]
     [ 0.1513  0.4048]
     [ 0.2103  0.4226]]

    This class also allows the user to run a spatial lag+error model with the
    extra feature of including non-spatial endogenous regressors. This means
    that, in addition to the spatial lag and error, we consider some of the
    variables on the right-hand side of the equation as endogenous and we
    instrument for this. As an example, we will include CRIME (crime rates) as
    endogenous and will instrument with DISCBD (distance to the CSB). We first
    need to read in the variables:


    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    And then we can run and explore the model analogously to the previous combo:

    >>> reg = GM_Combo_Hom(y, X, yd, q, w=w, A1='hom_sc', \
            name_ds='columbus')
    >>> betas = np.array([['CONSTANT'],['inc'],['crime'],['W_hoval'],['lambda']])
    >>> print(np.hstack((betas, np.around(np.hstack((reg.betas, np.sqrt(reg.vm.diagonal()).reshape(5,1))),5))))
    [['CONSTANT' '111.77057' '67.75191']
     ['inc' '-0.30974' '1.16656']
     ['crime' '-1.36043' '0.6841']
     ['W_hoval' '-0.52908' '0.84428']
     ['lambda' '0.60116' '0.18605']]

    """

    def __init__(
        self,
        y,
        x,
        yend=None,
        q=None,
        w=None,
        w_lags=1,
        slx_lags=0,
        slx_vars="All",
        lag_q=True,
        max_iter=1,
        epsilon=0.00001,
        A1="hom_sc",
        vm=False,
        name_y=None,
        name_x=None,
        name_yend=None,
        name_q=None,
        name_w=None,
        name_ds=None,
        latex=False,
        hard_bound=False,
    ):
        n = USER.check_arrays(y, x, yend, q)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        yend, q, name_yend, name_q = USER.check_endog([yend, q], [name_yend, name_q])
        x_constant, name_x, warn = USER.check_constant(x, name_x)
        name_x = USER.set_name_x(
            name_x, x_constant
        )  # initialize in case None, includes constant
        set_warn(self, warn)

        if slx_lags > 0:
            yend2, q2, wx = set_endog(
                y, x_constant[:, 1:], w, yend, q, w_lags, lag_q, slx_lags, slx_vars
            )
            x_constant = np.hstack((x_constant, wx))
        else:
            yend2, q2 = set_endog(y, x_constant[:, 1:], w, yend, q, w_lags, lag_q)

        # OLD
        # if slx_lags == 0:
        # yend2, q2 = set_endog(y, x_constant[:, 1:], w, yend, q, w_lags, lag_q)
        # else:
        # yend2, q2, wx = set_endog(y, x_constant[:, 1:], w, yend, q, w_lags, lag_q, slx_lags)
        # x_constant = np.hstack((x_constant, wx))

        BaseGM_Combo_Hom.__init__(
            self,
            y=y,
            x=x_constant,
            w=w.sparse,
            yend=yend2,
            q=q2,
            w_lags=w_lags,
            A1=A1,
            lag_q=lag_q,
            max_iter=max_iter,
            epsilon=epsilon,
            hard_bound=hard_bound,
        )
        self.rho = self.betas[-2]
        self.predy_e, self.e_pred, warn = sp_att(
            w, self.y, self.predy, yend2[:, -1].reshape(self.n, 1), self.rho
        )
        set_warn(self, warn)
        self.title = "SPATIALLY WEIGHTED 2SLS- GM-COMBO MODEL (HOM)"

        if slx_lags > 0:  # adjust for flexwx
            if isinstance(slx_vars, list):  # slx_vars has True,False
                if len(slx_vars) != x.shape[1]:
                    raise Exception("slx_vars incompatible with x column dimensions")
                else:  # use slx_vars to extract proper columns
                    workname = name_x[1:]
                    kx = len(workname)
                    vv = list(compress(workname, slx_vars))
                    name_x += USER.set_name_spatial_lags(vv, slx_lags)
                    wkx = slx_vars.count(True)
            else:
                kx = len(name_x) - 1
                wkx = kx
                name_x += USER.set_name_spatial_lags(
                    name_x[1:], slx_lags
                )  # exclude constant
            self.title += " WITH SLX (GNSM)"

        # OLD
        # if slx_lags > 0:
        #            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
        # name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)  # exclude constant
        # self.title += " WITH SLX (GNSM)"

        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        #        self.name_x = USER.set_name_x(name_x, x_constant)
        self.name_x = name_x  # constant already included
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_z.append("lambda")  # listing lambda last
        self.name_q = USER.set_name_q(name_q, q)

        if slx_lags > 0:  # need to remove all but last SLX variables from name_x
            self.name_x0 = []
            self.name_x0.append(self.name_x[0])  # constant
            if isinstance(slx_vars, list):  # boolean list passed
                # x variables that were not lagged
                self.name_x0.extend(
                    list(compress(self.name_x[1:], [not i for i in slx_vars]))
                )
                # last wkx variables
                self.name_x0.extend(self.name_x[-wkx:])

            else:
                okx = int(
                    (self.k - self.yend.shape[1] - 1) / (slx_lags + 1)
                )  # number of original exogenous vars

                self.name_x0.extend(self.name_x[-okx:])

            self.name_q.extend(
                USER.set_name_q_sp(self.name_x0, w_lags, self.name_q, lag_q)
            )

            # var_types = ['x'] * (kx + 1) + ['wx'] * kx * slx_lags + ['yend'] * (len(self.name_yend) - 1) + ['rho']
            var_types = (
                ["x"] * (kx + 1)
                + ["wx"] * wkx * slx_lags
                + ["yend"] * (len(self.name_yend) - 1)
                + ["rho", "lambda"]
            )
        else:
            self.name_q.extend(
                USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q)
            )
            var_types = (
                ["x"] * len(self.name_x)
                + ["yend"] * (len(self.name_yend) - 1)
                + ["rho", "lambda"]
            )

        # self.name_q.extend(USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q))
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        self.A1 = A1
        self.output = pd.DataFrame(self.name_z, columns=["var_names"])

        self.output["var_type"] = var_types

        # self.output['var_type'] = ['x'] * len(self.name_x) + ['yend'] * (len(self.name_yend) - 1) + ['rho', 'lambda']
        self.output["regime"], self.output["equation"] = (0, 0)
        self.other_top = _spat_pseudo_r2(self)
        self.other_top += _summary_iteration(self)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


# Functions


def moments_hom(w, wA1, wA2, u):
    """
    Compute G and g matrices for the spatial error model with homoscedasticity
    as in Anselin :cite:`Anselin2011` (2011).

    Parameters
    ----------

    w           : Sparse matrix
                  Spatial weights sparse matrix

    u           : array
                  Residuals. nx1 array assumed to be aligned with w

    Attributes
    ----------

    moments     : list
                  List of two arrays corresponding to the matrices 'G' and
                  'g', respectively.


    """
    n = w.shape[0]
    A1u = wA1 * u
    A2u = wA2 * u
    wu = w * u

    g1 = np.dot(u.T, A1u)
    g2 = np.dot(u.T, A2u)
    g = np.array([[g1][0][0], [g2][0][0]]) / n

    G11 = 2 * np.dot(wu.T * wA1, u)
    G12 = -np.dot(wu.T * wA1, wu)
    G21 = 2 * np.dot(wu.T * wA2, u)
    G22 = -np.dot(wu.T * wA2, wu)
    G = np.array([[G11[0][0], G12[0][0]], [G21[0][0], G22[0][0]]]) / n
    return [G, g]


def get_vc_hom(w, wA1, wA2, reg, lambdapar, z_s=None, for_omegaOLS=False):
    r"""
    VC matrix \psi of Spatial error with homoscedasticity. As in
    Anselin (2011) :cite:`Anselin2011` (p. 20)
    ...

    Parameters
    ----------
    w               :   Sparse matrix
                        Spatial weights sparse matrix
    reg             :   reg
                        Regression object
    lambdapar       :   float
                        Spatial parameter estimated in previous step of the
                        procedure
    z_s             :   array
                        optional argument for spatially filtered Z (to be
                        passed only if endogenous variables are present)
    for_omegaOLS    :   boolean
                        If True (default=False), it also returns P, needed
                        only in the computation of Omega

    Returns
    -------
    psi         : array
                  2x2 VC matrix
    a1          : array
                  nx1 vector a1. If z_s=None, a1 = 0.
    a2          : array
                  nx1 vector a2. If z_s=None, a2 = 0.
    p           : array
                  P matrix. If z_s=None or for_omegaOLS=False, p=0.

    """
    u_s = get_spFilter(w, lambdapar, reg.u)
    n = float(w.shape[0])
    sig2 = np.dot(u_s.T, u_s) / n
    mu3 = np.sum(u_s**3) / n
    mu4 = np.sum(u_s**4) / n

    tr11 = wA1 * wA1
    tr11 = np.sum(tr11.diagonal())
    tr12 = wA1 * (wA2 * 2)
    tr12 = np.sum(tr12.diagonal())
    tr22 = wA2 * wA2 * 2
    tr22 = np.sum(tr22.diagonal())
    vecd1 = np.array([wA1.diagonal()]).T

    psi11 = 2 * sig2**2 * tr11 + (mu4 - 3 * sig2**2) * np.dot(vecd1.T, vecd1)
    psi12 = sig2**2 * tr12
    psi22 = sig2**2 * tr22

    a1, a2, p = 0.0, 0.0, 0.0

    if for_omegaOLS:
        x_s = get_spFilter(w, lambdapar, reg.x)
        p = la.inv(spdot(x_s.T, x_s) / n)

    if (
        issubclass(type(z_s), np.ndarray)
        or issubclass(type(z_s), SP.csr.csr_matrix)
        or issubclass(type(z_s), SP.csc.csc_matrix)
    ):
        alpha1 = (-2 / n) * spdot(z_s.T, wA1 * u_s)
        alpha2 = (-2 / n) * spdot(z_s.T, wA2 * u_s)

        hth = spdot(reg.h.T, reg.h)
        hthni = la.inv(hth / n)
        htzsn = spdot(reg.h.T, z_s) / n
        p = spdot(hthni, htzsn)
        p = spdot(p, la.inv(spdot(htzsn.T, p)))
        hp = spdot(reg.h, p)
        a1 = spdot(hp, alpha1)
        a2 = spdot(hp, alpha2)

        psi11 = psi11 + sig2 * spdot(a1.T, a1) + 2 * mu3 * spdot(a1.T, vecd1)
        psi12 = psi12 + sig2 * spdot(a1.T, a2) + mu3 * spdot(a2.T, vecd1)  # 3rd term=0
        psi22 = psi22 + sig2 * spdot(a2.T, a2)  # 3rd&4th terms=0 bc vecd2=0

    psi = np.array([[psi11[0][0], psi12[0][0]], [psi12[0][0], psi22[0][0]]]) / n
    return psi, a1, a2, p


def get_omega_hom(w, wA1, wA2, reg, lamb, G):
    """
    Omega VC matrix for Hom models with endogenous variables computed as in
    Anselin (2011) :cite:`Anselin2011` (p. 21).
    ...

    Parameters
    ----------
    w       :   Sparse matrix
                Spatial weights sparse matrix
    reg     :   reg
                Regression object
    lamb    :   float
                Spatial parameter estimated in previous step of the
                procedure
    G       :   array
                Matrix 'G' of the moment equation

    Returns
    -------
    omega   :   array
                Omega matrix of VC of the model

    """
    n = float(w.shape[0])
    z_s = get_spFilter(w, lamb, reg.z)
    u_s = get_spFilter(w, lamb, reg.u)
    sig2 = np.dot(u_s.T, u_s) / n
    mu3 = np.sum(u_s**3) / n
    vecdA1 = np.array([wA1.diagonal()]).T
    psi, a1, a2, p = get_vc_hom(w, wA1, wA2, reg, lamb, z_s)
    j = np.dot(G, np.array([[1.0], [2 * lamb]]))
    psii = la.inv(psi)
    t2 = spdot(reg.h.T, np.hstack((a1, a2)))
    psiDL = (
        mu3 * spdot(reg.h.T, np.hstack((vecdA1, np.zeros((int(n), 1)))))
        + sig2 * spdot(reg.h.T, np.hstack((a1, a2)))
    ) / n

    oDD = spdot(la.inv(spdot(reg.h.T, reg.h)), spdot(reg.h.T, z_s))
    oDD = sig2 * la.inv(spdot(z_s.T, spdot(reg.h, oDD)))
    oLL = la.inv(spdot(j.T, spdot(psii, j))) / n
    oDL = spdot(spdot(spdot(p.T, psiDL), spdot(psii, j)), oLL)

    o_upper = np.hstack((oDD, oDL))
    o_lower = np.hstack((oDL.T, oLL))
    return np.vstack((o_upper, o_lower)), float(sig2)


def get_omega_hom_ols(w, wA1, wA2, reg, lamb, G):
    """
    Omega VC matrix for Hom models without endogenous variables (OLS) computed
    as in Anselin (2011) :cite:`Anselin2011`.
    ...

    Parameters
    ----------
    w       :   Sparse matrix
                Spatial weights sparse matrix
    reg     :   reg
                Regression object
    lamb    :   float
                Spatial parameter estimated in previous step of the
                procedure
    G       :   array
                Matrix 'G' of the moment equation

    Returns
    -------
    omega   :   array
                Omega matrix of VC of the model

    """
    n = float(w.shape[0])
    x_s = get_spFilter(w, lamb, reg.x)
    u_s = get_spFilter(w, lamb, reg.u)
    sig2 = np.dot(u_s.T, u_s) / n
    vecdA1 = np.array([wA1.diagonal()]).T
    psi, a1, a2, p = get_vc_hom(w, wA1, wA2, reg, lamb, for_omegaOLS=True)
    j = np.dot(G, np.array([[1.0], [2 * lamb]]))
    psii = la.inv(psi)

    oDD = sig2 * la.inv(spdot(x_s.T, x_s))
    oLL = la.inv(spdot(j.T, spdot(psii, j))) / n
    # oDL = np.zeros((oDD.shape[0], oLL.shape[1]))
    mu3 = np.sum(u_s**3) / n
    psiDL = (mu3 * spdot(reg.x.T, np.hstack((vecdA1, np.zeros((int(n), 1)))))) / n
    oDL = spdot(spdot(spdot(p.T, psiDL), spdot(psii, j)), oLL)

    o_upper = np.hstack((oDD, oDL))
    o_lower = np.hstack((oDL.T, oLL))
    return np.vstack((o_upper, o_lower)), float(sig2)


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
    y = np.array(db.by_col("HOVAL"))
    y = np.reshape(y, (49, 1))
    X = []
    X.append(db.by_col("INC"))
    X = np.array(X).T
    yd = []
    yd.append(db.by_col("CRIME"))
    yd = np.array(yd).T
    q = []
    q.append(db.by_col("DISCBD"))
    q = np.array(q).T

    w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    w.transform = "r"
    # reg = GM_Error_Hom(y, X, w=w, name_x=['inc'], name_y='hoval', name_ds='columbus', vm=True)
    # reg = GM_Endog_Error_Hom(y, X, yd, q, w=w, name_x=['inc'], name_y='hoval', name_yend=['crime'],
    #                         name_q=['discbd'], name_ds='columbus',vm=True)
    reg = GM_Combo_Hom(
        y,
        X,
        yd,
        q,
        w=w,
        name_x=["inc"],
        name_y="hoval",
        name_yend=["crime"],
        name_q=["discbd"],
        name_ds="columbus",
        vm=True,
    )

    print(reg.output)
    print(reg.summary)
