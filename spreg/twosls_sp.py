"""
Spatial Two Stages Least Squares
"""

__author__ = "Luc Anselin lanselin@gmail.com, David C. Folch david.folch@asu.edu"

import numpy as np
from . import twosls as TSLS
from . import user_output as USER
from .utils import set_endog, sp_att, set_warn
import pandas as pd
from .output import output, _spat_diag_out, _spat_pseudo_r2, _summary_impacts
from itertools import compress

__all__ = ["GM_Lag"]


class BaseGM_Lag(TSLS.BaseTSLS):
    """
    Spatial two stage least squares (S2SLS) (note: no consistency checks,
    diagnostics or constant added); Anselin (1988) [Anselin1988]_

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable; assumes the constant is
                   in column 0.
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note:
                   this should not contain any variables from x); cannot be
                   used in combination with h
    w            : Pysal weights matrix
                   Spatial weights matrix
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional
                   instruments (q).
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the Spatial Durbin type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
    robust       : string
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given.  If 'hac', then a
                   HAC consistent estimator of the variance-covariance
                   matrix is given. Default set to None.
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n.


    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    kstar        : integer
                   Number of endogenous variables.
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
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   H'H
    hthi         : float
                   (H'H)^-1
    varb         : array
                   (Z'H (H'H)^-1 H'Z)^-1
    zthhthi      : array
                   Z'H(H'H)^-1
    pfora1a2     : array
                   n(zthhthi)'varb

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> import spreg
    >>> np.set_printoptions(suppress=True) #prevent scientific format
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = 'r'
    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),'r')
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> # no non-spatial endogenous variables
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> reg = spreg.twosls_sp.BaseGM_Lag(y, X, w=w, w_lags=2)
    >>> reg.betas
    array([[45.30170561],
           [ 0.62088862],
           [-0.48072345],
           [ 0.02836221]])
    >>> spreg.se_betas(reg)
    array([17.91278862,  0.52486082,  0.1822815 ,  0.31740089])
    >>> reg = spreg.twosls_sp.BaseGM_Lag(y, X, w=w, w_lags=2, robust='white')
    >>> reg.betas
    array([[45.30170561],
           [ 0.62088862],
           [-0.48072345],
           [ 0.02836221]])
    >>> spreg.se_betas(reg)
    array([20.47077481,  0.50613931,  0.20138425,  0.38028295])
    >>> # instrument for HOVAL with DISCBD
    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("CRIME"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> reg = spreg.twosls_sp.BaseGM_Lag(y, X, w=w, yend=yd, q=q, w_lags=2)
    >>> reg.betas
    array([[100.79359082],
           [ -0.50215501],
           [ -1.14881711],
           [ -0.38235022]])
    >>> spreg.se_betas(reg)
    array([53.0829123 ,  1.02511494,  0.57589064,  0.59891744])

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
            robust=None,
            gwk=None,
            sig2n_k=False,
    ):
        


        if slx_lags > 0:
            yend2, q2, wx = set_endog(y, x[:, 1:], w, yend, q, w_lags, lag_q, slx_lags,slx_vars)
            x = np.hstack((x, wx))
        else:
            yend2, q2 = set_endog(y, x[:, 1:], w, yend, q, w_lags, lag_q)

        

        TSLS.BaseTSLS.__init__(
            self, y=y, x=x, yend=yend2, q=q2, robust=robust, gwk=gwk, sig2n_k=sig2n_k
        )


class GM_Lag(BaseGM_Lag):
    """
    Spatial two stage least squares (S2SLS) with results and diagnostics;
    Anselin (1988) :cite:`Anselin1988`

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
                   this should not contain any variables from x); cannot be
                   used in combination with h
    w            : pysal W object
                   Spatial weights object
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional
                   instruments (q).
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the Spatial Durbin type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
    robust       : string
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given.  If 'hac', then a
                   HAC consistent estimator of the variance-covariance
                   matrix is given. Default set to None.
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n.
    spat_diag    : boolean
                   If True, then compute Anselin-Kelejian test and Common Factor Hypothesis test (if applicable)
    spat_impacts : string or list
                   Include average direct impact (ADI), average indirect impact (AII),
                    and average total impact (ATI) in summary results.
                    Options are 'simple', 'full', 'power', 'all' or None.
                    See sputils.spmultiplier for more information.
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
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the bounds of -1 and 1.
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
    kstar        : integer
                   Number of endogenous variables.
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
    robust       : string
                   Adjustment for robust standard errors
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
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    ak_test      : tuple
                   Anselin-Kelejian test; tuple contains the pair (statistic,
                   p-value)
    cfh_test     : tuple
                   Common Factor Hypothesis test; tuple contains the pair (statistic,
                   p-value). Only when it applies (see specific documentation).
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_z       : list of strings
                   Names of exogenous and endogenous variables for use in
                   output
    name_q       : list of strings
                   Names of external instruments
    name_h       : list of strings
                   Names of all instruments used in ouput
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    title        : string
                   Name of the regression method used
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   :math:`H'H`
    hthi         : float
                   :math:`(H'H)^{-1}`
    varb         : array
                   :math:`(Z'H (H'H)^{-1} H'Z)^{-1}`
    zthhthi      : array
                   :math:`Z'H(H'H)^{-1}`
    pfora1a2     : array
                   n(zthhthi)'varb
    sp_multipliers: dict
                   Dictionary of spatial multipliers (if spat_impacts is not None)

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis. Since we will need some tests for our
    model, we also import the diagnostics module.

    >>> import numpy as np
    >>> import libpysal
    >>> import spreg

    Open data on Columbus neighborhood crime (49 areas) using libpysal.io.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),'r')

    Extract the HOVAL column (home value) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) and CRIME (crime rates) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in, but this can be overridden by passing
    constant=False.

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
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    This class runs a lag model, which means that includes the spatial lag of
    the dependent variable on the right-hand side of the equation. If we want
    to have the names of the variables printed in the
    output summary, we will have to pass them in as well, although this is
    optional. The default most basic model to be run would be:

    >>> from spreg import GM_Lag
    >>> np.set_printoptions(suppress=True) #prevent scientific format
    >>> reg=GM_Lag(y, X, w=w, w_lags=2, name_x=['inc', 'crime'], name_y='hoval', name_ds='columbus')
    >>> reg.betas
    array([[45.30170561],
           [ 0.62088862],
           [-0.48072345],
           [ 0.02836221]])

    Once the model is run, we can obtain the standard error of the coefficient
    estimates by calling the diagnostics module:

    >>> spreg.se_betas(reg)
    array([17.91278862,  0.52486082,  0.1822815 ,  0.31740089])

    But we can also run models that incorporates corrected standard errors
    following the White procedure. For that, we will have to include the
    optional parameter ``robust='white'``:

    >>> reg=GM_Lag(y, X, w=w, w_lags=2, robust='white', name_x=['inc', 'crime'], name_y='hoval', name_ds='columbus')
    >>> reg.betas
    array([[45.30170561],
           [ 0.62088862],
           [-0.48072345],
           [ 0.02836221]])

    And we can access the standard errors from the model object:

    >>> reg.std_err
    array([20.47077481,  0.50613931,  0.20138425,  0.38028295])

    The class is flexible enough to accomodate a spatial lag model that,
    besides the spatial lag of the dependent variable, includes other
    non-spatial endogenous regressors. As an example, we will assume that
    CRIME is actually endogenous and we decide to instrument for it with
    DISCBD (distance to the CBD). We reload the X including INC only and
    define CRIME as endogenous and DISCBD as instrument:

    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("CRIME"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))

    And we can run the model again:

    >>> reg=GM_Lag(y, X, w=w, yend=yd, q=q, w_lags=2, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')
    >>> reg.betas
    array([[100.79359082],
           [ -0.50215501],
           [ -1.14881711],
           [ -0.38235022]])

    Once the model is run, we can obtain the standard error of the coefficient
    estimates by calling the diagnostics module:

    >>> spreg.se_betas(reg)
    array([53.0829123 ,  1.02511494,  0.57589064,  0.59891744])

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
            slx_lags=0,
            slx_vars="All",
            robust=None,
            gwk=None,
            sig2n_k=False,
            spat_diag=True,
            spat_impacts="simple",
            vm=False,
            name_y=None,
            name_x=None,
            name_yend=None,
            name_q=None,
            name_w=None,
            name_gwk=None,
            name_ds=None,
            latex=False,
            hard_bound=False,
    ):

        n = USER.check_arrays(x, yend, q)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        USER.check_robust(robust, gwk)
        yend, q, name_yend, name_q = USER.check_endog([yend, q], [name_yend, name_q])
        spat_diag, warn = USER.check_spat_diag(spat_diag=spat_diag, w=w, robust=robust, slx_lags=slx_lags)
        set_warn(self, warn)
        x_constant, name_x, warn = USER.check_constant(x, name_x)
        set_warn(self, warn)
        name_x = USER.set_name_x(name_x, x_constant)  # need to check for None and set defaults

        # kx and wkx are used to replace complex calculation for output
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
                

        BaseGM_Lag.__init__(
            self,
            y=y,
            x=x_constant,
            w=w,
            yend=yend,
            q=q,
            w_lags=w_lags,
            slx_lags=slx_lags,
            slx_vars=slx_vars,
            robust=robust,
            gwk=gwk,
            lag_q=lag_q,
            sig2n_k=sig2n_k,
        )

        self.rho = self.betas[-1]
        self.predy_e, self.e_pred, warn = sp_att(
            w, self.y, self.predy, self.yend[:, -1].reshape(self.n, 1), self.rho, hard_bound=hard_bound
        )
        set_warn(self, warn)
        self.title = "SPATIAL TWO STAGE LEAST SQUARES"
        if slx_lags > 0:
            self.title += " WITH SLX (SPATIAL DURBIN MODEL)"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        #        self.name_x = USER.set_name_x(name_x, x_constant)   # name_x contains SLX terms for slx_lags > 0
        self.name_x = name_x  # already contains constant in new setup
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
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
                okx = int((self.k - self.kstar - 1) / (slx_lags + 1))  # number of original exogenous vars

                self.name_x0.extend(self.name_x[-okx:])

            self.name_q.extend(USER.set_name_q_sp(self.name_x0, w_lags, self.name_q, lag_q))

            #var_types = ['x'] * (kx + 1) + ['wx'] * kx * slx_lags + ['yend'] * (len(self.name_yend) - 1) + ['rho']
            var_types = ['x'] * (kx + 1) + ['wx'] * wkx * slx_lags + ['yend'] * (len(self.name_yend) - 1) + ['rho']
        else:
            self.name_q.extend(USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q))
            var_types = ['x'] * len(self.name_x) + ['yend'] * (len(self.name_yend) - 1) + ['rho']

        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.robust = USER.set_robust(robust)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        self.slx_lags = slx_lags
        self.slx_vars = slx_vars

        self.output = pd.DataFrame(self.name_x + self.name_yend, columns=['var_names'])
        self.output['var_type'] = var_types
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top = _spat_pseudo_r2(self)
        diag_out = None

        if spat_diag:
            diag_out = _spat_diag_out(self, w, 'yend')
        if spat_impacts:
            self.sp_multipliers, impacts_str = _summary_impacts(self, w, spat_impacts, slx_lags,slx_vars)
            try:
                diag_out += impacts_str
            except TypeError:
                diag_out = impacts_str
        output(reg=self, vm=vm, robust=robust, other_end=diag_out, latex=latex)


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
    yd_var = ["HOVAL"]
    yd = np.array([db.by_col(name) for name in yd_var]).T
    q_var = ["DISCBD"]
    q = np.array([db.by_col(name) for name in q_var]).T
    w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    w.transform = "r"
    model = GM_Lag(
        y,
        x,
        yd,
        q,
        w=w,
        spat_diag=True,
        name_y=y_var,
        name_x=x_var,
        name_yend=yd_var,
        name_q=q_var,
        name_ds="columbus",
        name_w="columbus.gal",
    )
    print(model.output)
    print(model.summary)
