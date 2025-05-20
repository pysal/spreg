"""
Spatial Two Stages Least Squares with Regimes
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu, David C. Folch david.folch@asu.edu"

import numpy as np
import pandas as pd
import multiprocessing as mp
from . import regimes as REGI
from . import user_output as USER
from .twosls_regimes import TSLS_Regimes, _optimal_weight
from .twosls import BaseTSLS
from .utils import set_endog, set_endog_sparse, sp_att, set_warn, sphstack, spdot, optim_k
from .robust import hac_multi
from .output import output, _spat_diag_out, _spat_pseudo_r2, _summary_impacts
from .skater_reg import Skater_reg
from .twosls_sp import BaseGM_Lag

class GM_Lag_Regimes(TSLS_Regimes, REGI.Regimes_Frame):

    """
    Spatial two stage least squares (S2SLS) with regimes;
    :cite:`Anselin1988`

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    regimes      : list or pandas.Series
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    yend         : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note:
                   this should not contain any variables from x); cannot be
                   used in combination with h
    constant_regi: string
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   * 'one': a vector of ones is appended to x and held constant across regimes.

                   * 'many': a vector of ones is appended to x and considered different per regime (default).
    cols2regi    : list, 'all'
                   Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all' (default), all the variables vary by regime.
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
    slx_vars     : either "all" (default) or list of booleans to select x variables
                   to be lagged        
    regime_lag_sep: boolean
                    If True (default), the spatial parameter for spatial lag is also
                    computed according to different regimes. If False,
                    the spatial parameter is fixed accross regimes.
                    Option valid only when regime_err_sep=True
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    robust       : string
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given.
                   If 'hac', then a HAC consistent estimator of the
                   variance-covariance matrix is given.
                   If 'ogmm', then Optimal GMM is used to estimate
                   betas and the variance-covariance matrix.
                   Default set to None.
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n.
    spat_impacts : string or list
                   Include average direct impact (ADI), average indirect impact (AII),
                    and average total impact (ATI) in summary results.
                    Options are 'simple', 'full', 'power', 'all' or None.
                    See sputils.spmultiplier for more information.
    spat_diag    : boolean
                   If True, then compute Anselin-Kelejian test and Common Factor Hypothesis test (if applicable)
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    cores        : boolean
                   Specifies if multiprocessing is to be used
                   Default: no multiprocessing, cores = False
                   Note: Multiprocessing may not work on all platforms.
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
    name_regimes : string
                   Name of regimes variable for use in output
    latex        : boolean
                   Specifies if summary is to be printed in latex format

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
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    kstar        : integer
                   Number of endogenous variables.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z            : array
                   nxk array of variables (combination of x and yend)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    h            : array
                   nxl array of instruments (combination of x and q)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    robust       : string
                   Adjustment for robust standard errors
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    std_err      : array
                   1xk array of standard errors of the betas
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    ak_test      : tuple
                   Anselin-Kelejian test; tuple contains the pair (statistic,
                   p-value)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
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
    name_regimes : string
                   Name of regimes variable for use in output
    title        : string
                   Name of the regression method used
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   :math:`H'H`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    hthi         : float
                   :math:`(H'H)^{-1}`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    varb         : array
                   :math:`(Z'H (H'H)^{-1} H'Z)^{-1}`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    zthhthi      : array
                   :math:`Z'H(H'H)^{-1}`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pfora1a2     : array
                   n(zthhthi)'varb
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sp_multipliers: dict
                   Dictionary of spatial multipliers (if spat_impacts is not None)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: string
                   Ignored if regimes=False. Constant option for regimes.
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   * 'one': a vector of ones is appended to x and held constant across regimes.

                   * 'many': a vector of ones is appended to x and considered different per regime.
    cols2regi    : list, 'all'
                   Ignored if regimes=False. Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all', all the variables vary by regime.
    regime_lag_sep: boolean
                    If True, the spatial parameter for spatial lag is also
                    computed according to different regimes. If False (default),
                    the spatial parameter is fixed accross regimes.
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    kr           : int
                   Number of variables/columns to be "regimized" or subject
                   to change by regime. These will result in one parameter
                   estimate by regime for each variable (i.e. nr parameters per
                   variable)
    kf           : int
                   Number of variables/columns to be considered fixed or
                   global across regimes and hence only obtain one parameter
                   estimate
    nr           : int
                   Number of different regimes in the 'regimes' list
    multi        : dictionary
                   Only available when multiple regressions are estimated,
                   i.e. when regime_err_sep=True and no variable is fixed
                   across regimes.
                   Contains all attributes of each individual regression

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import libpysal
    >>> from libpysal import examples

    Open data on NCOVR US County Homicides (3085 areas) using libpysal.io.open().
    This is the DBF associated with the NAT shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> db = libpysal.io.open(examples.get_path("NAT.dbf"),'r')

    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'HR90'
    >>> y = np.array([db.by_col(y_var)]).reshape(3085,1)

    Extract UE90 (unemployment rate) and PS90 (population structure) vectors from
    the DBF to be used as independent variables in the regression. Other variables
    can be inserted by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in.

    >>> x_var = ['PS90','UE90']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    The different regimes in this data are given according to the North and
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial lag model, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations. To do that, we can open an already existing gal file or
    create a new one. In this case, we will create one from ``NAT.shp``.

    >>> from libpysal import weights
    >>> w = weights.Rook.from_shapefile(examples.get_path("NAT.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    This class runs a lag model, which means that includes the spatial lag of
    the dependent variable on the right-hand side of the equation. If we want
    to have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> from spreg import GM_Lag_Regimes
    >>> model=GM_Lag_Regimes(y, x, regimes, w=w, regime_lag_sep=False, regime_err_sep=False, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT', name_w='NAT.shp')
    >>> model.betas
    array([[ 1.28897623],
           [ 0.79777722],
           [ 0.56366891],
           [ 8.73327838],
           [ 1.30433406],
           [ 0.62418643],
           [-0.39993716]])

    Once the model is run, we can have a summary of the output by typing:
    model.summary . Alternatively, we can obtain the standard error of
    the coefficient estimates by calling:

    >>> model.std_err
    array([0.38492902, 0.19106926, 0.06063249, 1.25607153, 0.36117334,
           0.092293  , 0.15116983])

    In the example above, all coefficients but the spatial lag vary
    according to the regime. It is also possible to have the spatial lag
    varying according to the regime, which effective will result in an
    independent spatial lag model estimated for each regime. To run these
    models, the argument regime_lag_sep must be set to True:

    >>> model=GM_Lag_Regimes(y, x, regimes, w=w, regime_lag_sep=True, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT', name_w='NAT.shp')
    >>> print(model.output)
        var_names coefficients   std_err   zt_stat      prob
    0  0_CONSTANT     1.365848  0.385177  3.546023  0.000391
    1      0_PS90     0.808757  0.206672   3.91325  0.000091
    2      0_UE90     0.569468  0.067703  8.411247       0.0
    3    0_W_HR90    -0.434244  0.177208 -2.450478  0.014267
    4  1_CONSTANT     7.907311  1.772336  4.461518  0.000008
    5      1_PS90     1.274657  0.368306  3.460869  0.000538
    6      1_UE90     0.601677  0.102102  5.892907       0.0
    7    1_W_HR90    -0.296034  0.226243 -1.308474  0.190712

    Alternatively, we can type: 'model.summary' to see the organized results output.
    The class is flexible enough to accomodate a spatial lag model that,
    besides the spatial lag of the dependent variable, includes other
    non-spatial endogenous regressors. As an example, we will add the endogenous
    variable RD90 (resource deprivation) and we decide to instrument for it with
    FP89 (families below poverty):

    >>> yd_var = ['RD90']
    >>> yd = np.array([db.by_col(name) for name in yd_var]).T
    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    And we can run the model again:

    >>> model = GM_Lag_Regimes(y, x, regimes, yend=yd, q=q, w=w, regime_lag_sep=False, regime_err_sep=False, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT', name_w='NAT.shp')
    >>> model.betas
    array([[ 3.42195202],
           [ 1.03311878],
           [ 0.14308741],
           [ 8.99740066],
           [ 1.91877758],
           [-0.32084816],
           [ 2.38918212],
           [ 3.67243761],
           [ 0.06959139]])

    Once the model is run, we can obtain the standard error of the coefficient
    estimates. Alternatively, we can have a summary of the output by typing:
    model.summary

    >>> model.std_err
    array([0.49529467, 0.18912143, 0.05157813, 0.92277557, 0.33711135,
           0.08993181, 0.33506177, 0.36381449, 0.07209498])
    """

    def __init__(
        self,
        y,
        x,
        regimes,
        yend=None,
        q=None,
        w=None,
        w_lags=1,
        slx_lags=0,
        slx_vars='all',
        lag_q=True,
        robust='white',
        gwk=None,
        sig2n_k=False,
        spat_diag=True,
        spat_impacts="simple",
        constant_regi="many",
        cols2regi="all",
        regime_lag_sep=False,
        regime_err_sep=False,
        cores=False,
        vm=False,
        name_y=None,
        name_x=None,
        name_yend=None,
        name_q=None,
        name_regimes=None,
        name_w=None,
        name_gwk=None,
        name_ds=None,
        latex=False,
        hard_bound=False,
    ):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        yend, q, name_yend, name_q = USER.check_endog([yend, q], [name_yend, name_q])
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        USER.check_robust(robust, gwk)
        if regime_lag_sep and not regime_err_sep:
            set_warn(self, "regime_err_sep set to True when regime_lag_sep=True.")                
            regime_err_sep = True
        if regime_err_sep and not regime_lag_sep:
            set_warn(self, "Groupwise heteroskedasticity is not currently available for this method,\n so regime_err_sep has been set to False.")                
            regime_err_sep = False
        if robust == "hac":
            if regime_err_sep:
                set_warn(
                    self,
                    "Error by regimes is not available for HAC estimation. The error by regimes has been disabled for this model.",
                )
                regime_err_sep = False
        spat_diag, warn = USER.check_spat_diag(spat_diag=spat_diag, w=w, robust=robust, slx_lags=slx_lags)
        set_warn(self, warn)
        x_constant, name_x, warn = USER.check_constant(x, name_x, just_rem=True)
        set_warn(self, warn)
        name_x = USER.set_name_x(name_x, x_constant, constant=True)
        name_y = USER.set_name_y(name_y)
        name_yend = USER.set_name_yend(name_yend, yend)
        name_q = USER.set_name_q(name_q, q)

        regimes, name_regimes = USER.check_reg_list(regimes, name_regimes, n)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.constant_regi = constant_regi
        if slx_lags > 0:
            yend2, q2, wx = set_endog(y, x_constant, w, yend, q, w_lags, lag_q, slx_lags, slx_vars=slx_vars)
            set_warn(self,"WX is computed using the complete W, i.e. not trimmed by regimes.")
            x_constant = np.hstack((x_constant, wx))
            name_slx = USER.set_name_spatial_lags(name_x, slx_lags, slx_vars=slx_vars)
            iter_slx = iter(name_slx) 
            name_q_temp = [next(iter_slx) if keep else name for name, keep in zip(name_x, slx_vars)]
            name_q.extend(USER.set_name_q_sp(name_q_temp, w_lags, name_q, lag_q, force_all=True))
            name_x += name_slx
            if cols2regi == 'all':
                cols2regi = REGI.check_cols2regi(
                    constant_regi, cols2regi, x_constant, yend=yend2, add_cons=False)[0:-1]
            else:
                cols2regi = REGI.check_cols2regi(
                    constant_regi, cols2regi, x_constant, yend=yend2, add_cons=False)
        else:
            name_q.extend(USER.set_name_q_sp(name_x, w_lags, name_q, lag_q, force_all=True))
            yend2, q2 = yend, q
            cols2regi = REGI.check_cols2regi(
                constant_regi, cols2regi, x_constant, yend=yend2, add_cons=False)
        self.n = x_constant.shape[0]
        self.cols2regi = cols2regi
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, self.n, x_constant.shape[1])
        if regime_err_sep == True and robust == "hac":
            set_warn(
                self,
                "Error by regimes is incompatible with HAC estimation for Spatial Lag models. Hence, error and lag by regimes have been disabled for this model.",
            )
            regime_err_sep = False
            regime_lag_sep = False
        self.regime_err_sep = regime_err_sep
        self.regime_lag_sep = regime_lag_sep

        if regime_lag_sep == True:
            cols2regi += [True]
            w_i, regi_ids, warn = REGI.w_regimes(
                w,
                regimes,
                self.regimes_set,
                transform=True,
                get_ids=True,
                min_n=len(cols2regi) + 1,
            )
            set_warn(self, warn)
        else:
            cols2regi += [False]

        if (
            regime_err_sep == True
            and set(cols2regi) == set([True])
            and constant_regi == "many"
        ):
            self.y = y
            self.GM_Lag_Regimes_Multi(
                y,
                x_constant,
                w_i,
                w,
                regi_ids,
                yend=yend2,
                q=q2,
                w_lags=w_lags,
                slx_lags=slx_lags,
                lag_q=lag_q,
                cores=cores,
                robust=robust,
                gwk=gwk,
                sig2n_k=sig2n_k,
                cols2regi=cols2regi,
                spat_impacts=spat_impacts,
                spat_diag=spat_diag,
                vm=vm,
                name_y=name_y,
                name_x=name_x,
                name_yend=name_yend,
                name_q=name_q,
                name_regimes=self.name_regimes,
                name_w=name_w,
                name_gwk=name_gwk,
                name_ds=name_ds,
                latex=latex,
                hard_bound=hard_bound,
            )
        else:
            if regime_lag_sep == True:
                w = REGI.w_regimes_union(w, w_i, self.regimes_set)
            if slx_lags == 0:
                yend2, q2 = set_endog(y, x_constant, w, yend2, q2, w_lags, lag_q)
            name_yend.append(USER.set_name_yend_sp(name_y))
            TSLS_Regimes.__init__(
                self,
                y=y,
                x=x_constant,
                yend=yend2,
                q=q2,
                regimes=regimes,
                w=w,
                robust=robust,
                gwk=gwk,
                sig2n_k=sig2n_k,
                spat_diag=spat_diag,
                vm=vm,
                constant_regi=constant_regi,
                cols2regi=cols2regi,
                regime_err_sep=regime_err_sep,
                name_y=name_y,
                name_x=name_x,
                name_yend=name_yend,
                name_q=name_q,
                name_regimes=name_regimes,
                name_w=name_w,
                name_gwk=name_gwk,
                name_ds=name_ds,
                summ=False,
            )

            if regime_lag_sep:  # not currently available.
                self.sp_att_reg(w_i, regi_ids, yend2[:, -1].reshape(self.n, 1))
            else:
                self.rho = self.betas[-1]
                self.output.iat[-1, self.output.columns.get_loc('var_type')] = 'rho'
                self.predy_e, self.e_pred, warn = sp_att(
                    w, self.y, self.predy, yend2[:, -1].reshape(self.n, 1), self.rho, hard_bound=hard_bound)
                set_warn(self, warn)
            self.regime_lag_sep = regime_lag_sep
            self.title = "SPATIAL " + self.title
            if slx_lags > 0:
                for m in self.regimes_set:
                    r_output = self.output[(self.output['regime'] == str(m)) & (self.output['var_type'].isin(['x', 'o']))]
                    wx_index = r_output.index[-((len(r_output)-1)//(slx_lags+1)) * slx_lags:]
                    self.output.loc[wx_index, 'var_type'] = 'wx'
                self.title = " SPATIAL 2SLS WITH SLX (SPATIAL DURBIN MODEL) - REGIMES"
            top_diag = _spat_pseudo_r2(self)
            try:
                self.other_top = top_diag + self.other_top
            except:
                self.other_top = top_diag
            self.slx_lags = slx_lags
            diag_out = None
            if spat_diag:
                diag_out = _spat_diag_out(self, w, 'yend')
            if spat_impacts:
                self.sp_multipliers, impacts_str = _summary_impacts(self, w, spat_impacts, slx_lags, regimes=True)
                try:
                    diag_out += impacts_str
                except TypeError:
                    diag_out = impacts_str
            output(reg=self, vm=vm, robust=robust, other_end=diag_out, latex=latex)

    def GM_Lag_Regimes_Multi(
        self,
        y,
        x,
        w_i,
        w,
        regi_ids,
        cores=False,
        yend=None,
        q=None,
        w_lags=1,
        slx_lags=0,
        lag_q=True,
        robust=None,
        gwk=None,
        sig2n_k=False,
        cols2regi="all",
        spat_impacts=False,
        spat_diag=False,
        vm=False,
        name_y=None,
        name_x=None,
        name_yend=None,
        name_q=None,
        name_regimes=None,
        name_w=None,
        name_gwk=None,
        name_ds=None,
        latex=False,
        hard_bound=False,
    ):
        #        pool = mp.Pool(cores)
        self.name_ds = USER.set_name_ds(name_ds)
        name_yend.append(USER.set_name_yend_sp(name_y))
        self.name_w = USER.set_name_w(name_w, w_i)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        results_p = {}
        """
        for r in self.regimes_set:
            w_r = w_i[r].sparse
            if system() == 'Windows':
                is_win = True
                results_p[r] = _work(*(y,x,regi_ids,r,yend,q,w_r,w_lags,lag_q,robust,sig2n_k,self.name_ds,name_y,name_x,name_yend,name_q,self.name_w,name_regimes))
            else:                
                results_p[r] = pool.apply_async(_work,args=(y,x,regi_ids,r,yend,q,w_r,w_lags,lag_q,robust,sig2n_k,self.name_ds,name_y,name_x,name_yend,name_q,self.name_w,name_regimes, ))
                is_win = False
        """
        x_constant, name_x = REGI.check_const_regi(self, x, name_x, regi_ids)
        self.name_x_r = name_x
        for r in self.regimes_set:
            w_r = w_i[r].sparse
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(
                    _work,
                    args=(
                        y,
                        x_constant,
                        regi_ids,
                        r,
                        yend,
                        q,
                        w_r,
                        w_lags,
                        slx_lags,
                        lag_q,
                        robust,
                        sig2n_k,
                        self.name_ds,
                        name_y,
                        name_x,
                        name_yend,
                        name_q,
                        self.name_w,
                        name_regimes,
                    ),
                )
            else:
                results_p[r] = _work(
                    *(
                        y,
                        x_constant,
                        regi_ids,
                        r,
                        yend,
                        q,
                        w_r,
                        w_lags,
                        slx_lags,
                        lag_q,
                        robust,
                        sig2n_k,
                        self.name_ds,
                        name_y,
                        name_x,
                        name_yend,
                        name_q,
                        self.name_w,
                        name_regimes,
                    )
                )

        self.kryd = 0
        self.kr = len(cols2regi) + 1
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.name_x_r = name_x + name_yend
        self.name_regimes = name_regimes
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * self.kr, 1), float)
        self.u = np.zeros((self.n, 1), float)
        self.predy = np.zeros((self.n, 1), float)
        self.predy_e = np.zeros((self.n, 1), float)
        self.e_pred = np.zeros((self.n, 1), float)
        """
        if not is_win:
            pool.close()
            pool.join()
        """
        if cores:
            pool.close()
            pool.join()
        results = {}
        (
            self.name_y,
            self.name_x,
            self.name_yend,
            self.name_q,
            self.name_z,
            self.name_h,
        ) = ([], [], [], [], [], [])
        counter = 0
        self.output = pd.DataFrame(columns=['var_names', 'var_type', 'regime', 'equation'])
        for r in self.regimes_set:
            """
            if is_win:
                results[r] = results_p[r]
            else:
                results[r] = results_p[r].get()
            """
            if not cores:
                results[r] = results_p[r]
            else:
                results[r] = results_p[r].get()
            results[r].predy_e, results[r].e_pred, warn = sp_att(
                w_i[r],
                results[r].y,
                results[r].predy,
                results[r].yend[:, -1].reshape(results[r].n, 1),
                results[r].rho, hard_bound=hard_bound
            )
            set_warn(results[r], warn)
            results[r].w = w_i[r]
            self.vm[
                (counter * self.kr) : ((counter + 1) * self.kr),
                (counter * self.kr) : ((counter + 1) * self.kr),
            ] = results[r].vm
            self.betas[
                (counter * self.kr) : ((counter + 1) * self.kr),
            ] = results[r].betas
            self.u[
                regi_ids[r],
            ] = results[r].u
            self.predy[
                regi_ids[r],
            ] = results[r].predy
            self.predy_e[
                regi_ids[r],
            ] = results[r].predy_e
            self.e_pred[
                regi_ids[r],
            ] = results[r].e_pred
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            self.name_yend += results[r].name_yend
            self.name_q += results[r].name_q
            self.name_z += results[r].name_z
            self.name_h += results[r].name_h
            if r == self.regimes_set[0]:
                self.hac_var = np.zeros((self.n, results[r].h.shape[1]), float)
            self.hac_var[
                regi_ids[r],
            ] = results[r].h
            try:
                results[r].other_top = self.other_top
            except:
                results[r].other_top = ""
            results[r].other_top += _spat_pseudo_r2(results[r])
            results[r].other_mid = ""
            if slx_lags > 0:
                kx = (results[r].k - results[r].kstar - 1) // (slx_lags + 1)
                var_types = ['o'] + ['x']*kx + ['wx'] * kx * slx_lags + ['yend'] * (len(results[r].name_yend) - 1) + ['rho']
            else:
                var_types = ['o'] + ['x'] * (len(results[r].name_x)-1) + ['yend'] * (len(results[r].name_yend)-1) + ['rho']
            results[r].output = pd.DataFrame({'var_names': results[r].name_x + results[r].name_yend,
                                                                'var_type': var_types,
                                                                'regime': r, 'equation': r})
            self.output = pd.concat([self.output, results[r].output], ignore_index=True)
            if spat_diag:
                results[r].other_mid += _spat_diag_out(results[r], results[r].w, 'yend')
            if spat_impacts:
                results[r].sp_multipliers, impacts_str = _summary_impacts(results[r], results[r].w, spat_impacts, slx_lags)
                results[r].other_mid += impacts_str
            counter += 1
        self.multi = results
        if robust == "hac":
            hac_multi(self, gwk, constant=True)
        if robust == "ogmm":
            set_warn(
                self,
                "Residuals treated as homoskedastic for the purpose of diagnostics.",
            )
        self.chow = REGI.Chow(self)
        #if spat_diag:
        #   self._get_spat_diag_props(y, x, w, yend, q, w_lags, lag_q)
        output(reg=self, vm=vm, robust=robust, other_end=False, latex=latex)

    def sp_att_reg(self, w_i, regi_ids, wy):
        predy_e_r, e_pred_r = {}, {}
        self.predy_e = np.zeros((self.n, 1), float)
        self.e_pred = np.zeros((self.n, 1), float)
        counter = 1
        for r in self.regimes_set:
            self.rho = self.betas[
                (self.kr - self.kryd) * self.nr
                + self.kf
                - (self.yend.shape[1] - self.nr * self.kryd)
                + self.kryd * counter
                - 1
            ]
            self.predy_e[regi_ids[r],], self.e_pred[regi_ids[r],], warn = sp_att(
                w_i[r],
                self.y[regi_ids[r]],
                self.predy[regi_ids[r]],
                wy[regi_ids[r]],
                self.rho,
            )
            counter += 1

    def _get_spat_diag_props(self, y, x, w, yend, q, w_lags, lag_q):
        self._cache = {}
        yend, q = set_endog(y, x[:, 1:], w, yend, q, w_lags, lag_q)
        # x = USER.check_constant(x)
        x = REGI.regimeX_setup(x, self.regimes, [True] * x.shape[1], self.regimes_set)
        self.z = sphstack(
            x,
            REGI.regimeX_setup(
                yend,
                self.regimes,
                [True] * (yend.shape[1] - 1) + [False],
                self.regimes_set,
            ),
        )
        self.h = sphstack(
            x,
            REGI.regimeX_setup(q, self.regimes, [True] * q.shape[1], self.regimes_set),
        )
        hthi = np.linalg.inv(spdot(self.h.T, self.h))
        zth = spdot(self.z.T, self.h)
        self.varb = np.linalg.inv(spdot(spdot(zth, hthi), zth.T))


def _work(
    y,
    x,
    regi_ids,
    r,
    yend,
    q,
    w_r,
    w_lags,
    slx_lags,
    lag_q,
    robust,
    sig2n_k,
    name_ds,
    name_y,
    name_x,
    name_yend,
    name_q,
    name_w,
    name_regimes,
):
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    if yend is not None:
        yend_r = yend[regi_ids[r]]
    else:
        yend_r = yend
    if q is not None:
        q_r = q[regi_ids[r]]
    else:
        q_r = q
    if slx_lags == 0:
        yend_r, q_r = set_endog_sparse(y_r, x_r[:, 1:], w_r, yend_r, q_r, w_lags, lag_q)
        title = "SPATIAL TWO STAGE LEAST SQUARES ESTIMATION - REGIME %s" % r
    else:
        title = "SPATIAL 2SLS WITH SLX (SPATIAL DURBIN MODEL) - REGIME %s" % r
    # x_constant = USER.check_constant(x_r)
    if robust == "hac" or robust == "ogmm":
        robust2 = None
    else:
        robust2 = robust
    model = BaseTSLS(y_r, x_r, yend_r, q_r, robust=robust2, sig2n_k=sig2n_k)
    model.title = title
    if robust == "ogmm":
        _optimal_weight(model, sig2n_k, warn=False)
    model.rho = model.betas[-1]
    model.robust = USER.set_robust(robust)
    model.name_ds = name_ds
    model.name_y = "%s_%s" % (str(r), name_y)
    model.name_x = ["%s_%s" % (str(r), i) for i in name_x]
    model.name_yend = ["%s_%s" % (str(r), i) for i in name_yend]
    model.name_z = model.name_x + model.name_yend
    model.name_q = ["%s_%s" % (str(r), i) for i in name_q]
    model.name_h = model.name_x + model.name_q
    model.name_w = name_w
    model.name_regimes = name_regimes
    model.slx_lags = slx_lags
    return model


class GM_Lag_Endog_Regimes(GM_Lag_Regimes):

    """
    Spatial two stage least squares (S2SLS) with endogenous regimes. 
    Based on the function skater_reg as shown in :cite:`Anselin2021`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object (required if running spatial
                   diagnostics)
    n_clusters   : int
                   Number of clusters to be used in the endogenous regimes.
                   If None (default), the number of clusters will be chosen
                   according to the function utils.optim_k using a method 
                   adapted from Mojena (1977)'s Rule Two
    quorum       : int
                   Minimum number of observations in a cluster to be considered
                   Must be at least larger than the number of variables in x
                   Default value is 30 or 10*k, whichever is larger.
    trace        : boolean
                   Sets  whether to store intermediate results of the clustering
                   Hard-coded to True if n_clusters is None
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regimes variable for use in output
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    **kwargs     : additional keyword arguments depending on the specific model

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
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    kstar        : integer
                   Number of endogenous variables.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z            : array
                   nxk array of variables (combination of x and yend)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    h            : array
                   nxl array of instruments (combination of x and q)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    robust       : string
                   Adjustment for robust standard errors
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    std_err      : array
                   1xk array of standard errors of the betas
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    ak_test      : tuple
                   Anselin-Kelejian test; tuple contains the pair (statistic,
                   p-value)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
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
    name_regimes : string
                   Name of regimes variable for use in output
    title        : string
                   Name of the regression method used
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   :math:`H'H`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    hthi         : float
                   :math:`(H'H)^{-1}`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    varb         : array
                   :math:`(Z'H (H'H)^{-1} H'Z)^{-1}`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    zthhthi      : array
                   :math:`Z'H(H'H)^{-1}`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pfora1a2     : array
                   n(zthhthi)'varb
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sp_multipliers: dict
                   Dictionary of spatial multipliers (if spat_impacts is not None)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: string
                   Ignored if regimes=False. Constant option for regimes.
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   * 'one': a vector of ones is appended to x and held constant across regimes.

                   * 'many': a vector of ones is appended to x and considered different per regime.
    cols2regi    : list, 'all'
                   Ignored if regimes=False. Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all', all the variables vary by regime.
    regime_lag_sep: boolean
                    If True, the spatial parameter for spatial lag is also
                    computed according to different regimes. If False (default),
                    the spatial parameter is fixed accross regimes.
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    kr           : int
                   Number of variables/columns to be "regimized" or subject
                   to change by regime. These will result in one parameter
                   estimate by regime for each variable (i.e. nr parameters per
                   variable)
    kf           : int
                   Number of variables/columns to be considered fixed or
                   global across regimes and hence only obtain one parameter
                   estimate
    nr           : int
                   Number of different regimes in the 'regimes' list
    multi        : dictionary
                   Only available when multiple regressions are estimated,
                   i.e. when regime_err_sep=True and no variable is fixed
                   across regimes.
                   Contains all attributes of each individual regression
    SSR          : list
                   list with the total sum of squared residuals for the model 
                   considering all regimes for each of steps of number of regimes
                   considered, starting with the solution with 2 regimes. 
    clusters     : int
                   Number of clusters considered in the endogenous regimes
    _trace       : list
                   List of dictionaries with the clustering results for each
                   number of clusters tested. Only available if n_clusters is
                   None or trace=True.
    Examples
    --------
    >>> import libpysal
    >>> import numpy as np
    >>> np.set_printoptions(legacy='1.25') #to avoid printing issues with numpy floats
    >>> import geopandas as gpd
    >>> from spreg import GM_Lag_Endog_Regimes

    Open data on Baltimore house sales price and characteristics in Baltimore
    from libpysal examples using geopandas.

    >>> db = gpd.read_file(libpysal.examples.get_path('baltim.shp'))

    We will create a weights matrix based on contiguity.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = "r"

    For this example, we will use the 'PRICE' column as the dependent variable, and
    the 'NROOM', 'AGE', and 'SQFT' columns as independent variables.
    At this point, we will let the model choose the number of clusters.

    >>> reg = GM_Lag_Endog_Regimes(y=db['PRICE'], x=db[['NROOM','AGE','SQFT']], w=w, name_w="baltim_q.gal")
    
    The function `print(reg.summary)` can be used to visualize the results of the regression.

    Alternatively, we can check individual attributes:
    >>> reg.betas
    array([[ 6.20932938],
           [ 4.25581944],
           [-0.1468118 ],
           [ 0.40893082],
           [ 5.01866492],
           [ 4.84994184],
           [-0.55425337],
           [ 1.04577632],
           [ 0.05155043]])
    >>> reg.SSR
    [59784.06769835169, 56858.621800274515]
    >>> reg.clusters
    array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
           1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
           1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int32)

    We  will now set the number of clusters to 2 and run the regression again.

    >>> reg = GM_Lag_Endog_Regimes(y=db['PRICE'], x=db[['NROOM','AGE','SQFT']], w=w, n_clusters=2, name_w="baltim_q.gal")

    The function `print(reg.summary)` can be used to visualize the results of the regression.

    Alternatively, we can check individual attributes as before:
    >>> reg.betas
    array([[ 6.20932938],
           [ 4.25581944],
           [-0.1468118 ],
           [ 0.40893082],
           [ 5.01866492],
           [ 4.84994184],
           [-0.55425337],
           [ 1.04577632],
           [ 0.05155043]])
    >>> reg.SSR
    [59784.06769835169]
    >>> reg.clusters
    array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
           1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
           1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int32)

    """


    def __init__(
        self, y, x, w, n_clusters=None, quorum=-1, trace=True, name_y=None, name_x=None,
        constant_regi='many', cols2regi='all', regime_err_sep=False, **kwargs):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True)
        x_constant, name_x, warn = USER.check_constant(x, name_x, just_rem=True)
        set_warn(self, warn)
        warn = USER.check_regi_args(constant_regi, cols2regi, regime_err_sep, err_flag=False)
        set_warn(self, warn)
        # Standardize the variables
        x_std = (x_constant - np.mean(x_constant, axis=0)) / np.std(x_constant, axis=0)

        if quorum < 0:
            quorum = np.max([(x.shape[1]+1)*10, 30])
        
        if not n_clusters:
            n_clusters_opt = x_constant.shape[0]*0.70//quorum
            if n_clusters_opt < 2:
                raise ValueError(
                    "The combination of the values of `N` and `quorum` is not compatible with regimes estimation.")
            sk_reg_results = Skater_reg().fit(n_clusters_opt, w, x_std, {'reg':BaseGM_Lag,'y':y,'x':x_constant,'w':w}, quorum=quorum, trace=True)
            n_clusters = optim_k([sk_reg_results._trace[i][1][2] for i in range(1, len(sk_reg_results._trace))])
            self.clusters = sk_reg_results._trace[n_clusters-1][0]
            self.score = sk_reg_results._trace[n_clusters-1][1][2]
        else:
            try:
                # Call the Skater_reg method based on GM_Lag
                sk_reg_results = Skater_reg().fit(n_clusters, w, x_std, {'reg':BaseGM_Lag,'y':y,'x':x_constant,'w':w}, quorum=quorum, trace=trace)
                self.clusters = sk_reg_results.current_labels_
                self.score = sk_reg_results._trace[-1][1][2]
            except Exception as e:
                if str(e) == "one or more input arrays have more columns than rows":
                    raise ValueError("One or more input ended up with more variables than observations. Please check your setting for `quorum`.")
                else:
                    print("An error occurred:", e)

        self._trace = sk_reg_results._trace
        self.SSR = [self._trace[i][1][2] for i in range(1, len(self._trace))]
        GM_Lag_Regimes.__init__(self, y, x, regimes=self.clusters, w=w, name_y=name_y, name_x=name_x, name_regimes='Skater_reg',
                                constant_regi='many', cols2regi='all', regime_err_sep=False, **kwargs)


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
    from libpysal import examples

    db = libpysal.io.open(examples.get_path("columbus.dbf"), "r")
    y_var = "CRIME"
    y = np.array([db.by_col(y_var)]).reshape(49, 1)
    x_var = ["INC"]
    x = np.array([db.by_col(name) for name in x_var]).T
    yd_var = ["HOVAL"]
    yd = np.array([db.by_col(name) for name in yd_var]).T
    q_var = ["DISCBD"]
    q = np.array([db.by_col(name) for name in q_var]).T
    r_var = "NSA"
    regimes = db.by_col(r_var)
    w = libpysal.weights.Queen.from_shapefile(
        libpysal.examples.get_path("columbus.shp")
    )
    w.transform = "r"
    model = GM_Lag_Regimes(
        y,
        x,
        regimes,
        yend=yd,
        q=q,
        w=w,
        constant_regi="many",
        spat_diag=True,
        sig2n_k=False,
        lag_q=True,
        name_y=y_var,
        name_x=x_var,
        name_yend=yd_var,
        name_q=q_var,
        name_regimes=r_var,
        name_ds="columbus",
        name_w="columbus.gal",
        regime_err_sep=True,
        regime_lag_sep = True,
        robust="white",
    )
    print(model.output)
    print(model.summary)
