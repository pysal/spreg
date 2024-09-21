"""
Spatial Error with Heteroskedasticity family of models
"""

__author__ = "Luc Anselin lanselin@gmail.com, \
        Pedro V. Amaral pedro.amaral@asu.edu, \
        Daniel Arribas-Bel darribas@asu.edu, \
        David C. Folch david.folch@asu.edu \
        Ran Wei rwei5@asu.edu"

import numpy as np
import numpy.linalg as la
from . import ols as OLS
from . import user_output as USER
from . import twosls as TSLS
from . import utils as UTILS
from .utils import RegressionPropsY, spdot, set_endog, sphstack, set_warn, get_lags
from scipy import sparse as SP
from libpysal.weights.spatial_lag import lag_spatial
import pandas as pd
from .output import output, _summary_iteration, _spat_pseudo_r2
from itertools import compress

__all__ = ["GM_Error_Het", "GM_Endog_Error_Het", "GM_Combo_Het"]


class BaseGM_Error_Het(RegressionPropsY):

    """
    GMM method for a spatial error model with heteroskedasticity (note: no
    consistency checks, diagnostics or constant added); based on
    :cite:`Arraiz2010`, following :cite:`Anselin2011`.

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
                   Maximum number of iterations of steps 2a and 2b from
                   :cite:`Arraiz2010`. Note: epsilon provides an additional
                   stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from :cite:`Arraiz2010`. Note: max_iter provides
                   an additional stop condition.
    step1c       : boolean
                   If True, then include Step 1c from :cite:`Arraiz2010`.
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
    xtx          : float
                   X'X

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
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = 'r'
    >>> reg = spreg.error_sp_het.BaseGM_Error_Het(y, X, w.sparse, step1c=True)
    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[47.9963 11.479 ]
     [ 0.7105  0.3681]
     [-0.5588  0.1616]
     [ 0.4118  0.168 ]]
    """

    def __init__(self, y, x, w, max_iter=1, epsilon=0.00001, step1c=False, hard_bound=False):

        self.step1c = step1c
        # 1a. OLS --> \tilde{betas}
        ols = OLS.BaseOLS(y=y, x=x)
        self.x, self.y, self.n, self.k, self.xtx = ols.x, ols.y, ols.n, ols.k, ols.xtx
        wA1 = UTILS.get_A1_het(w)

        # 1b. GMM --> \tilde{\lambda1}
        moments = UTILS._moments2eqs(wA1, w, ols.u)
        lambda1 = UTILS.optim_moments(moments)

        if step1c:
            # 1c. GMM --> \tilde{\lambda2}
            sigma = get_psi_sigma(w, ols.u, lambda1)
            vc1 = get_vc_het(w, wA1, sigma)
            lambda2 = UTILS.optim_moments(moments, vc1)
        else:
            lambda2 = lambda1

        lambda_old = lambda2

        self.iteration, eps = 0, 1
        while self.iteration < max_iter and eps > epsilon:
            # 2a. reg -->\hat{betas}
            xs = UTILS.get_spFilter(w, lambda_old, self.x)
            ys = UTILS.get_spFilter(w, lambda_old, self.y)
            ols_s = OLS.BaseOLS(y=ys, x=xs)
            self.predy = spdot(self.x, ols_s.betas)
            self.u = self.y - self.predy

            # 2b. GMM --> \hat{\lambda}
            sigma_i = get_psi_sigma(w, self.u, lambda_old)
            vc_i = get_vc_het(w, wA1, sigma_i)
            moments_i = UTILS._moments2eqs(wA1, w, self.u)
            lambda3 = UTILS.optim_moments(moments_i, vc_i)
            eps = abs(lambda3 - lambda_old)
            lambda_old = lambda3
            self.iteration += 1

        self.iter_stop = UTILS.iter_msg(self.iteration, max_iter)
        if hard_bound:
            if abs(lambda3) >= 0.99:
                raise Exception("Spatial error parameter was outside the bounds of -0.99 and 0.99")
        else:
            if abs(lambda3) >= 0.99:
                set_warn(self, "Spatial error parameter was outside the bounds of -0.99 and 0.99")

        sigma = get_psi_sigma(w, self.u, lambda3)
        vc3 = get_vc_het(w, wA1, sigma)
        self.vm = get_vm_het(moments_i[0], lambda3, self, w, vc3)
        self.betas = np.vstack((ols_s.betas, lambda3))
        self.e_filtered = self.u - lambda3 * w * self.u
        self._cache = {}


class GM_Error_Het(BaseGM_Error_Het):
    """
    GMM method for a spatial error model with heteroskedasticity, with results
    and diagnostics; based on :cite:`Arraiz2010`, following
    :cite:`Anselin2011`.

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
                   Note: epsilon provides an additional
                   stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from :cite:`Arraiz2010`. Note: max_iter provides
                   an additional stop condition.
    step1c       : boolean
                   If True, then include Step 1c from :cite:`Arraiz2010`.
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
                   Number of iterations of steps 2a and 2b from :cite:`Arraiz2010`.
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    vm           : array
                   Variance covariance matrix (kxk)
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

    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> from spreg import GM_Error_Het
    >>> reg = GM_Error_Het(y, X, w=w, step1c=True, name_y='home value', name_x=['income', 'crime'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. This class offers an error model that explicitly accounts
    for heteroskedasticity and that unlike the models from
    ``spreg.error_sp``, it allows for inference on the spatial
    parameter.

    >>> print(reg.name_x)
    ['CONSTANT', 'income', 'crime', 'lambda']

    Hence, we find the same number of betas as of standard errors,
    which we calculate taking the square root of the diagonal of the
    variance-covariance matrix:

    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[47.9963 11.479 ]
     [ 0.7105  0.3681]
     [-0.5588  0.1616]
     [ 0.4118  0.168 ]]

    Alternatively, we can have a summary of the output by typing:
    print(reg.summary)

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
        step1c=False,
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
        name_x = USER.set_name_x(name_x, x_constant)  # initialize in case None, includes constant
        set_warn(self, warn)
        self.title = "GM SPATIALLY WEIGHTED LEAST SQUARES (HET)"

        if slx_lags >0:
            #lag_x = get_lags(w, x_constant[:, 1:], slx_lags)
            #x_constant = np.hstack((x_constant, lag_x))
#            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
            #name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags) # exclude constant

            x_constant,name_x = USER.flex_wx(w,x=x_constant,name_x=name_x,constant=True,
                                             slx_lags=slx_lags,slx_vars=slx_vars)

            self.title += " WITH SLX (SLX-Error)"

        # OLD
        #if slx_lags >0:
            #lag_x = get_lags(w, x_constant[:, 1:], slx_lags)
            #x_constant = np.hstack((x_constant, lag_x))
#            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
            #name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)  # no constant
            #self.title += " WITH SLX (SLX-Error)"

        BaseGM_Error_Het.__init__(
            self,
            y=y,
            x=x_constant,
            w=w.sparse,
            max_iter=max_iter,
            step1c=step1c,
            epsilon=epsilon,
            hard_bound = hard_bound
        )


        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
#        self.name_x = USER.set_name_x(name_x, x_constant)
        self.name_x = name_x  # constant already included
        self.name_x.append("lambda")
        self.name_w = USER.set_name_w(name_w, w)
        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['x'] * (len(self.name_x)-1) + ['lambda']
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top = _summary_iteration(self)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


class BaseGM_Endog_Error_Het(RegressionPropsY):

    """
    GMM method for a spatial error model with heteroskedasticity and
    endogenous variables (note: no consistency checks, diagnostics or constant
    added); based on :cite:`Arraiz2010`, following :cite:`Anselin2011`.

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
    step1c       : boolean
                   If True, then include Step 1c from :cite:`Arraiz2010`.
    inv_method   : string
                   If "power_exp", then compute inverse using the power
                   expansion. If "true_inv", then compute the true inverse.
                   Note that true_inv will fail for large n.
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
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = 'r'
    >>> reg = spreg.error_sp_het.BaseGM_Endog_Error_Het(y, X, yd, q, w=w.sparse, step1c=True)
    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[55.3971 28.8901]
     [ 0.4656  0.7731]
     [-0.6704  0.468 ]
     [ 0.4114  0.1777]]
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
        step1c=False,
        inv_method="power_exp",
        hard_bound=False,
    ):

        self.step1c = step1c
        # 1a. reg --> \tilde{betas}
        tsls = TSLS.BaseTSLS(y=y, x=x, yend=yend, q=q)

        if abs(tsls.betas[-1]) <= 0.9:
            pass
        else:
            if tsls.betas[-1] < -0.9:
                tsls.betas[-1] = -0.9
            else:
                tsls.betas[-1] = 0.9
            tsls.u = tsls.y - spdot(tsls.z, tsls.betas)

        self.x, self.z, self.h, self.y = tsls.x, tsls.z, tsls.h, tsls.y
        self.yend, self.q, self.n, self.k, self.hth = (
            tsls.yend,
            tsls.q,
            tsls.n,
            tsls.k,
            tsls.hth,
        )
        wA1 = UTILS.get_A1_het(w)

        # 1b. GMM --> \tilde{\lambda1}
        moments = UTILS._moments2eqs(wA1, w, tsls.u)
        lambda1 = UTILS.optim_moments(moments)

        if step1c:
            # 1c. GMM --> \tilde{\lambda2}
            self.u = tsls.u
            zs = UTILS.get_spFilter(w, lambda1, self.z)
            vc1 = get_vc_het_tsls(
                w, wA1, self, lambda1, tsls.pfora1a2, zs, inv_method, filt=False
            )
            lambda2 = UTILS.optim_moments(moments, vc1)
        else:
            lambda2 = lambda1

        # Forcing the 1st step lambda to be in the range [-0.9, 0.9] to avoid perfect collinearity in step 2 in case of SLX-Error or GNS models
        #if lambda2 > 0.9:
        #    lambda_old = 0.9
        #elif lambda2 < -0.9:
        #    lambda_old = -0.9
        #else:
        lambda_old = lambda2

        self.iteration, eps = 0, 1
        while self.iteration < max_iter and eps > epsilon:
            # 2a. reg -->\hat{betas}
            xs = UTILS.get_spFilter(w, lambda_old, self.x)
            ys = UTILS.get_spFilter(w, lambda_old, self.y)
            yend_s = UTILS.get_spFilter(w, lambda_old, self.yend)
            tsls_s = TSLS.BaseTSLS(ys, xs, yend_s, h=self.h)

            if abs(tsls_s.betas[-1]) <= 0.9:
                pass
            else:
                if tsls_s.betas[-1] < -0.9:
                    tsls_s.betas[-1] = -0.9
                else:
                    tsls_s.betas[-1] = 0.9
                tsls_s.u = tsls_s.y - spdot(tsls_s.z, tsls_s.betas)

            self.predy = spdot(self.z, tsls_s.betas)
            self.u = self.y - self.predy

            # 2b. GMM --> \hat{\lambda}
            vc2 = get_vc_het_tsls(
                w,
                wA1,
                self,
                lambda_old,
                tsls_s.pfora1a2,
                sphstack(xs, yend_s),
                inv_method,
            )
            moments_i = UTILS._moments2eqs(wA1, w, self.u)
            lambda3 = UTILS.optim_moments(moments_i, vc2)

            #if abs(lambda3) <= 0.9:
            #    pass
            #elif lambda3 > 0.9:
            #    lambda3 = 0.9
            #elif lambda3 < -0.9:
            #    lambda3 = -0.9

            eps = abs(lambda3 - lambda_old)
            lambda_old = lambda3
            self.iteration += 1

        self.iter_stop = UTILS.iter_msg(self.iteration, max_iter)
        if hard_bound:
            if abs(lambda3) >= 0.99:
                raise Exception("Spatial error parameter was outside the bounds of -0.99 and 0.99")
            if abs(tsls_s.betas[-1]) >= 0.99:
                raise Exception("Spatial lag parameter was outside the bounds of -0.99 and 0.99")
        else:
            if abs(lambda3) >= 0.99:
                set_warn(self, "Spatial error parameter was outside the bounds of -0.99 and 0.99")
            if abs(tsls_s.betas[-1]) >= 0.99:
                set_warn(self, "Spatial lag parameter was outside the bounds of -0.99 and 0.99")

        zs = UTILS.get_spFilter(w, lambda3, self.z)
        P = get_P_hat(self, tsls.hthi, zs)
        vc3 = get_vc_het_tsls(w, wA1, self, lambda3, P, zs, inv_method, save_a1a2=True)
        self.vm = get_Omega_GS2SLS(w, lambda3, self, moments_i[0], vc3, P)
        self.betas = np.vstack((tsls_s.betas, lambda3))
        self.e_filtered = self.u - lambda3 * w * self.u
        self._cache = {}


class GM_Endog_Error_Het(BaseGM_Endog_Error_Het):

    """
    GMM method for a spatial error model with heteroskedasticity and
    endogenous variables, with results and diagnostics; based on
    :cite:`Arraiz2010`, following :cite:`Anselin2011`.

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
                   :cite:`Arraiz2010`. Note: epsilon provides an additional
                   stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from :cite:`Arraiz2010`. Note: max_iter provides
                   an additional stop condition.
    step1c       : boolean
                   If True, then include Step 1c from :cite:`Arraiz2010`.
    inv_method   : string
                   If "power_exp", then compute inverse using the power
                   expansion. If "true_inv", then compute the true inverse.
                   Note that true_inv will fail for large n.
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
    >>> from spreg import GM_Endog_Error_Het

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

    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables (exogenous and endogenous), the
    instruments and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> reg = GM_Endog_Error_Het(y, X, yd, q, w=w, step1c=True, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. This class offers an error model that explicitly accounts
    for heteroskedasticity and that unlike the models from
    ``spreg.error_sp``, it allows for inference on the spatial
    parameter. Hence, we find the same number of betas as of standard errors,
    which we calculate taking the square root of the diagonal of the
    variance-covariance matrix:

    >>> print(reg.name_z)
    ['CONSTANT', 'inc', 'crime', 'lambda']
    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[55.3971 28.8901]
     [ 0.4656  0.7731]
     [-0.6704  0.468 ]
     [ 0.4114  0.1777]]

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
        step1c=False,
        inv_method="power_exp",
        vm=False,
        name_y=None,
        name_x=None,
        name_yend=None,
        name_q=None,
        name_w=None,
        name_ds=None,
        latex=False,
        hard_bound=False
    ):

        n = USER.check_arrays(y, x, yend, q)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        yend, q, name_yend, name_q = USER.check_endog([yend, q], [name_yend, name_q])
        x_constant, name_x, warn = USER.check_constant(x, name_x)
        name_x = USER.set_name_x(name_x, x_constant)  # initialize in case None, includes constant
        set_warn(self, warn)
        self.title = "GM SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES (HET)"

        if slx_lags >0:

            x_constant,name_x = USER.flex_wx(w,x=x_constant,name_x=name_x,constant=True,
                                             slx_lags=slx_lags,slx_vars=slx_vars)

            self.title += " WITH SLX (SLX-Error)"   

        # OLD
        #if slx_lags > 0:
            #lag_x = get_lags(w, x_constant[:, 1:], slx_lags)
            #x_constant = np.hstack((x_constant, lag_x))
            #name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)    # no constant
            #self.title += " WITH SLX (SLX-Error)"


        BaseGM_Endog_Error_Het.__init__(
            self,
            y=y,
            x=x_constant,
            yend=yend,
            q=q,
            w=w.sparse,
            max_iter=max_iter,
            step1c=step1c,
            epsilon=epsilon,
            inv_method=inv_method,
            hard_bound=hard_bound,
        )
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
#        self.name_x = USER.set_name_x(name_x, x_constant)
        self.name_x = name_x  # constant already included
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append("lambda")  # listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        self.output = pd.DataFrame(self.name_z,
                                   columns=['var_names'])
        self.output['var_type'] = ['x'] * len(self.name_x) + ['yend'] * len(self.name_yend) + ['lambda']
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top = _summary_iteration(self)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


class BaseGM_Combo_Het(BaseGM_Endog_Error_Het):

    """
    GMM method for a spatial lag and error model with heteroskedasticity and
    endogenous variables (note: no consistency checks, diagnostics or constant
    added); based on :cite:`Arraiz2010`, following :cite:`Anselin2011`.

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
                   Maximum number of iterations of steps 2a and 2b from
                   :cite:`Arraiz2010`. Note: epsilon provides an additional
                   stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from :cite:`Arraiz2010`. Note: max_iter provides
                   an additional stop condition.
    step1c       : boolean
                   If True, then include Step 1c from :cite:`Arraiz2010`.
    inv_method   : string
                   If "power_exp", then compute inverse using the power
                   expansion. If "true_inv", then compute the true inverse.
                   Note that true_inv will fail for large n.
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

    >>> reg = spreg.error_sp_het.BaseGM_Combo_Het(y, X, yend=yd2, q=q2, w=w.sparse, step1c=True)
    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[ 9.9753 14.1435]
     [ 1.5742  0.374 ]
     [ 0.1535  0.3978]
     [ 0.2103  0.3924]]

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
    >>> reg = spreg.error_sp_het.BaseGM_Combo_Het(y, X, yd2, q2, w=w.sparse, step1c=True)
    >>> betas = np.array([['CONSTANT'],['inc'],['crime'],['lag_hoval'],['lambda']])
    >>> print(np.hstack((betas, np.around(np.hstack((reg.betas, np.sqrt(reg.vm.diagonal()).reshape(5,1))),5))))
    [['CONSTANT' '113.91292' '64.38815']
     ['inc' '-0.34822' '1.18219']
     ['crime' '-1.35656' '0.72482']
     ['lag_hoval' '-0.57657' '0.75856']
     ['lambda' '0.65608' '0.15719']]
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
        step1c=False,
        inv_method="power_exp",
        hard_bound=False,
    ):

        BaseGM_Endog_Error_Het.__init__(
            self,
            y=y,
            x=x,
            w=w,
            yend=yend,
            q=q,
            max_iter=max_iter,
            step1c=step1c,
            epsilon=epsilon,
            inv_method=inv_method,
            hard_bound=hard_bound,
        )


class GM_Combo_Het(BaseGM_Combo_Het):

    """
    GMM method for a spatial lag and error model with heteroskedasticity and
    endogenous variables, with results and diagnostics; based on
    :cite:`Arraiz2010`, following :cite:`Anselin2011`.

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
                   Spatial weights object (always needed)
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
    step1c       : boolean
                   If True, then include Step 1c from :cite:`Arraiz2010`.
    inv_method   : string
                   If "power_exp", then compute inverse using the power
                   expansion. If "true_inv", then compute the true inverse.
                   Note that true_inv will fail for large n.
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
    >>> from spreg import GM_Combo_Het

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

    The Combo class runs an SARAR model, that is a spatial lag+error model.
    In this case we will run a simple version of that, where we have the
    spatial effects as well as exogenous variables. Since it is a spatial
    model, we have to pass in the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> reg = GM_Combo_Het(y, X, w=w, step1c=True, name_y='hoval', name_x=['income'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. This class offers an error model that explicitly accounts
    for heteroskedasticity and that unlike the models from
    ``spreg.error_sp``, it allows for inference on the spatial
    parameter. Hence, we find the same number of betas as of standard errors,
    which we calculate taking the square root of the diagonal of the
    variance-covariance matrix:

    >>> print(reg.name_z)
    ['CONSTANT', 'income', 'W_hoval', 'lambda']
    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[ 9.9753 14.1435]
     [ 1.5742  0.374 ]
     [ 0.1535  0.3978]
     [ 0.2103  0.3924]]

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

    >>> reg = GM_Combo_Het(y, X, yd, q, w=w, step1c=True, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')
    >>> print(reg.name_z)
    ['CONSTANT', 'inc', 'crime', 'W_hoval', 'lambda']
    >>> print(np.round(reg.betas,4))
    [[113.9129]
     [ -0.3482]
     [ -1.3566]
     [ -0.5766]
     [  0.6561]]

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
        step1c=False,
        inv_method="power_exp",
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
        name_x = USER.set_name_x(name_x, x_constant)  # initialize in case None, includes constant
        set_warn(self, warn)

        if slx_lags > 0:
            yend2, q2, wx = set_endog(y, x_constant[:, 1:], w, yend, q, w_lags, lag_q, slx_lags,slx_vars)
            x_constant = np.hstack((x_constant, wx))
        else:
            yend2, q2 = set_endog(y, x_constant[:, 1:], w, yend, q, w_lags, lag_q)

        # OLS
        #if slx_lags == 0:
            #yend2, q2 = set_endog(y, x_constant[:, 1:], w, yend, q, w_lags, lag_q)
        #else:
            #yend2, q2, wx = set_endog(y, x_constant[:, 1:], w, yend, q, w_lags, lag_q, slx_lags)
            #x_constant = np.hstack((x_constant, wx))

        BaseGM_Combo_Het.__init__(
            self,
            y=y,
            x=x_constant,
            yend=yend2,
            q=q2,
            w=w.sparse,
            w_lags=w_lags,
            max_iter=max_iter,
            step1c=step1c,
            lag_q=lag_q,
            epsilon=epsilon,
            inv_method=inv_method,
            hard_bound=hard_bound,
        )
        self.rho = self.betas[-2]
        self.predy_e, self.e_pred, warn = UTILS.sp_att(
            w, self.y, self.predy, yend2[:, -1].reshape(self.n, 1), self.rho
        )
        UTILS.set_warn(self, warn)
        self.title = "SPATIALLY WEIGHTED 2SLS- GM-COMBO MODEL (HET)"

        if slx_lags > 0:  # adjust for flexwx
            if (isinstance(slx_vars,list)):     # slx_vars has True,False
                if len(slx_vars) != x.shape[1] :
                    raise Exception("slx_vars incompatible with x column dimensions")
                else:  # use slx_vars to extract proper columns
                    workname = name_x[1:]
                    kx = len(workname)
                    vv = list(compress(workname,slx_vars))
                    name_x += USER.set_name_spatial_lags(vv, slx_lags)
                    wkx = slx_vars.count(True)
            else:
                kx = len(name_x) - 1
                wkx = kx
                name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)  # exclude constant
            self.title += " WITH SLX (GNSM)"

        # OLD
        #if slx_lags > 0:
#            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
            #name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)   # no constant
            #self.title += " WITH SLX (GNSM)"

        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
#        self.name_x = USER.set_name_x(name_x, x_constant)
        self.name_x = name_x   # constant already included
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_z.append("lambda")  # listing lambda last
        self.name_q = USER.set_name_q(name_q, q)

        if slx_lags > 0:  # need to remove all but last SLX variables from name_x
            self.name_x0 = []
            self.name_x0.append(self.name_x[0])  # constant
            if (isinstance(slx_vars,list)):   # boolean list passed
                # x variables that were not lagged
                self.name_x0.extend(list(compress(self.name_x[1:],[not i for i in slx_vars])))
                # last wkx variables
                self.name_x0.extend(self.name_x[-wkx:])


            else:
                okx = int((self.k - self.yend.shape[1] - 1) / (slx_lags + 1))  # number of original exogenous vars

                self.name_x0.extend(self.name_x[-okx:])

            self.name_q.extend(USER.set_name_q_sp(self.name_x0, w_lags, self.name_q, lag_q))

            #var_types = ['x'] * (kx + 1) + ['wx'] * kx * slx_lags + ['yend'] * (len(self.name_yend) - 1) + ['rho']
            var_types = ['x'] * (kx + 1) + ['wx'] * wkx * slx_lags + ['yend'] * (len(self.name_yend) - 1) + ['rho','lambda']
        else:
            self.name_q.extend(USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q))
            var_types = ['x'] * len(self.name_x) + ['yend'] * (len(self.name_yend) - 1) + ['rho','lambda']



        #self.name_q.extend(USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q))
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        self.output = pd.DataFrame(self.name_z,
                                   columns=['var_names'])
        
        self.output['var_type'] = var_types

        #self.output['var_type'] = ['x'] * len(self.name_x) + ['yend'] * (len(self.name_yend)-1) + ['rho', 'lambda']

        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top = _spat_pseudo_r2(self)
        self.other_top += _summary_iteration(self)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


# Functions


def get_psi_sigma(w, u, lamb):
    """
    Computes the Sigma matrix needed to compute Psi

    Parameters
    ----------
    w           : Sparse matrix
                  Spatial weights sparse matrix
    u           : array
                  nx1 vector of residuals
    lamb        : float
                  Lambda

    """

    e = (u - lamb * (w * u)) ** 2
    E = SP.dia_matrix((e.flat, 0), shape=(w.shape[0], w.shape[0]))
    return E.tocsr()


def get_vc_het(w, wA1, E):
    r"""
    Computes the VC matrix Psi based on lambda as in Arraiz et al :cite:`Arraiz2010`:

    ..math::

        \tilde{Psi} = \left(\begin{array}{c c}
                            \psi_{11} & \psi_{12} \\
                            \psi_{21} & \psi_{22} \\
                      \end{array} \right)

    NOTE: psi12=psi21

    ...

    Parameters
    ----------

    w           : Sparse matrix
                  Spatial weights sparse matrix

    E           : sparse matrix
                  Sigma

    Returns
    -------

    Psi         : array
                  2x2 array with estimator of the variance-covariance matrix

    """
    aPatE = 2 * wA1 * E
    wPwtE = (w + w.T) * E

    psi11 = aPatE * aPatE
    psi12 = aPatE * wPwtE
    psi22 = wPwtE * wPwtE
    psi = list(map(np.sum, [psi11.diagonal(), psi12.diagonal(), psi22.diagonal()]))
    return np.array([[psi[0], psi[1]], [psi[1], psi[2]]]) / (2.0 * w.shape[0])


def get_vm_het(G, lamb, reg, w, psi):
    """
    Computes the variance-covariance matrix Omega as in Arraiz et al
    :cite:`Arraiz2010`

    Parameters
    ----------

    G           : array
                  G from moments equations

    lamb        : float
                  Final lambda from spHetErr estimation

    reg         : regression object
                  output instance from a regression model

    u           : array
                  nx1 vector of residuals

    w           : Sparse matrix
                  Spatial weights sparse matrix

    psi         : array
                  2x2 array with the variance-covariance matrix of the moment equations

    Returns
    -------

    vm          : array
                  (k+1)x(k+1) array with the variance-covariance matrix of the parameters

    """

    J = np.dot(G, np.array([[1], [2 * lamb]]))
    Zs = UTILS.get_spFilter(w, lamb, reg.x)
    ZstEZs = spdot((Zs.T * get_psi_sigma(w, reg.u, lamb)), Zs)
    ZsZsi = la.inv(spdot(Zs.T, Zs))
    omega11 = w.shape[0] * np.dot(np.dot(ZsZsi, ZstEZs), ZsZsi)
    omega22 = la.inv(np.dot(np.dot(J.T, la.inv(psi)), J))
    zero = np.zeros((reg.k, 1), float)
    vm = (
        np.vstack((np.hstack((omega11, zero)), np.hstack((zero.T, omega22))))
        / w.shape[0]
    )
    return vm


def get_P_hat(reg, hthi, zf):
    """
    P_hat from Appendix B, used for a1 a2, using filtered Z
    """
    htzf = spdot(reg.h.T, zf)
    P1 = spdot(hthi, htzf)
    P2 = spdot(htzf.T, P1)
    P2i = la.inv(P2)
    return reg.n * np.dot(P1, P2i)


def get_a1a2(w, wA1, reg, lambdapar, P, zs, inv_method, filt):
    """
    Computes the a1 in psi assuming residuals come from original regression.
    :cite:`Anselin2011`

    Parameters
    ----------
    w           : Sparse matrix
                  Spatial weights sparse matrix
    reg         : TSLS
                  Two stage least quare regression instance
    lambdapar   : float
                  Spatial autoregressive parameter

    Returns
    -------
    [a1, a2]    : list
                  a1 and a2 are two nx1 array in psi equation

    """
    us = UTILS.get_spFilter(w, lambdapar, reg.u)
    alpha1 = (-2.0 / w.shape[0]) * (np.dot(spdot(zs.T, wA1), us))
    alpha2 = (-1.0 / w.shape[0]) * (np.dot(spdot(zs.T, (w + w.T)), us))
    a1 = np.dot(spdot(reg.h, P), alpha1)
    a2 = np.dot(spdot(reg.h, P), alpha2)
    if not filt:
        a1 = UTILS.inverse_prod(
            w, a1, lambdapar, post_multiply=True, inv_method=inv_method
        ).T
        a2 = UTILS.inverse_prod(
            w, a2, lambdapar, post_multiply=True, inv_method=inv_method
        ).T
    return [a1, a2]


def get_vc_het_tsls(
    w, wA1, reg, lambdapar, P, zs, inv_method, filt=True, save_a1a2=False
):

    sigma = get_psi_sigma(w, reg.u, lambdapar)
    vc1 = get_vc_het(w, wA1, sigma)
    a1, a2 = get_a1a2(w, wA1, reg, lambdapar, P, zs, inv_method, filt)
    a1s = a1.T * sigma
    a2s = a2.T * sigma
    psi11 = float(np.dot(a1s, a1))
    psi12 = float(np.dot(a1s, a2))
    psi21 = float(np.dot(a2s, a1))
    psi22 = float(np.dot(a2s, a2))
    psi0 = np.array([[psi11, psi12], [psi21, psi22]]) / w.shape[0]
    if save_a1a2:
        psi = (vc1 + psi0, a1, a2)
    else:
        psi = vc1 + psi0
    return psi


def get_Omega_GS2SLS(w, lamb, reg, G, psi, P):
    """
    Computes the variance-covariance matrix for GS2SLS:

    Parameters
    ----------
    w           : Sparse matrix
                  Spatial weights sparse matrix
    lamb        : float
                  Spatial autoregressive parameter
    reg         : GSTSLS
                  Generalized Spatial two stage least quare regression instance
    G           : array
                  Moments
    psi         : array
                  Weighting matrix

    Returns
    -------
    omega       : array
                  (k+1)x(k+1)
    """
    psi, a1, a2 = psi
    sigma = get_psi_sigma(w, reg.u, lamb)
    psi_dd_1 = (1.0 / w.shape[0]) * reg.h.T * sigma
    psi_dd = spdot(psi_dd_1, reg.h)
    psi_dl = spdot(psi_dd_1, np.hstack((a1, a2)))
    psi_o = np.hstack((np.vstack((psi_dd, psi_dl.T)), np.vstack((psi_dl, psi))))
    psii = la.inv(psi)

    j = np.dot(G, np.array([[1.0], [2 * lamb]]))
    jtpsii = np.dot(j.T, psii)
    jtpsiij = np.dot(jtpsii, j)
    jtpsiiji = la.inv(jtpsiij)
    omega_1 = np.dot(jtpsiiji, jtpsii)
    omega_2 = np.dot(np.dot(psii, j), jtpsiiji)
    om_1_s = omega_1.shape
    om_2_s = omega_2.shape
    p_s = P.shape
    omega_left = np.hstack(
        (
            np.vstack((P.T, np.zeros((om_1_s[0], p_s[0])))),
            np.vstack((np.zeros((p_s[1], om_1_s[1])), omega_1)),
        )
    )
    omega_right = np.hstack(
        (
            np.vstack((P, np.zeros((om_2_s[0], p_s[1])))),
            np.vstack((np.zeros((p_s[0], om_2_s[1])), omega_2)),
        )
    )
    omega = np.dot(np.dot(omega_left, psi_o), omega_right)
    return omega / w.shape[0]


def _test():
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    _test()
    import numpy as np
    import libpysal

    db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    y = np.array(db.by_col("HOVAL"))
    y = np.reshape(y, (49,1))
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
    w.transform = 'r'
    # reg = GM_Error_Het(y, X, w=w, name_x=['inc'], name_y='hoval', name_ds='columbus', vm=True)
    # reg = GM_Endog_Error_Het(y, X, yd, q, w=w, name_x=['inc'], name_y='hoval', name_yend=['crime'],
    #                         name_q=['discbd'], name_ds='columbus',vm=True)
    reg = GM_Combo_Het(y, X, yd, q, w=w, step1c=True, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus',
                       vm=True)
    print(reg.output)
    print(reg.summary)