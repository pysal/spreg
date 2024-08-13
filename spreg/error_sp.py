"""
Spatial Error Models module
"""

__author__ = "Luc Anselin lanselin@gmail.com, \
        Daniel Arribas-Bel darribas@asu.edu, \
        Pedro V. Amaral pedro.amaral@asu.edu"

import numpy as np
from numpy import linalg as la
from . import ols as OLS
from .utils import set_endog, sp_att, optim_moments, get_spFilter, get_lags, spdot, RegressionPropsY, set_warn
from . import twosls as TSLS
from . import user_output as USER
import pandas as pd
from .output import output, _spat_pseudo_r2
from .error_sp_het import GM_Error_Het, GM_Endog_Error_Het, GM_Combo_Het
from .error_sp_hom import GM_Error_Hom, GM_Endog_Error_Hom, GM_Combo_Hom
from itertools import compress


__all__ = ["GMM_Error", "GM_Error", "GM_Endog_Error", "GM_Combo"]


class BaseGM_Error(RegressionPropsY):

    """
    GMM method for a spatial error model (note: no consistency checks
    diagnostics or constant added); based on Kelejian and Prucha
    (1998, 1999) :cite:`Kelejian1998` :cite:`Kelejian1999`.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : Sparse matrix
                   Spatial weights sparse matrix
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
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations

    Examples
    --------

    >>> import libpysal
    >>> import numpy as np
    >>> import spreg
    >>> dbf = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array([dbf.by_col('HOVAL')]).T
    >>> x = np.array([dbf.by_col('INC'), dbf.by_col('CRIME')]).T
    >>> x = np.hstack((np.ones(y.shape),x))
    >>> w = libpysal.io.open(libpysal.examples.get_path("columbus.gal"), 'r').read()
    >>> w.transform='r'
    >>> model = spreg.error_sp.BaseGM_Error(y, x, w=w.sparse)
    >>> np.around(model.betas, decimals=4)
    array([[47.6946],
           [ 0.7105],
           [-0.5505],
           [ 0.3257]])
    """

    def __init__(self, y, x, w, hard_bound=False):

        # 1a. OLS --> \tilde{betas}
        ols = OLS.BaseOLS(y=y, x=x)
        self.n, self.k = ols.x.shape
        self.x = ols.x
        self.y = ols.y

        # 1b. GMM --> \tilde{\lambda1}
        moments = _momentsGM_Error(w, ols.u)
        lambda1 = optim_moments(moments, hard_bound=hard_bound)

        # 2a. OLS -->\hat{betas}
        xs = get_spFilter(w, lambda1, self.x)
        ys = get_spFilter(w, lambda1, self.y)
        ols2 = OLS.BaseOLS(y=ys, x=xs)

        # Output
        self.predy = spdot(self.x, ols2.betas)
        self.u = y - self.predy
        self.betas = np.vstack((ols2.betas, np.array([[lambda1]])))
        self.sig2 = ols2.sig2n
        self.e_filtered = self.u - lambda1 * w * self.u

        self.vm = self.sig2 * ols2.xtxi
        se_betas = np.sqrt(self.vm.diagonal())
        self._cache = {}


class GM_Error(BaseGM_Error):

    """
    GMM method for a spatial error model, with results and diagnostics; based
    on Kelejian and Prucha (1998, 1999) :cite:`Kelejian1998` :cite:`Kelejian1999`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object (always needed)
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX-Error type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
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
                   autoregressive parameter is outside maximum/minimum bounds.
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

    >>> import libpysal
    >>> import numpy as np
    >>> from spreg import GM_Error

    Open data on Columbus neighborhood crime (49 areas) using libpysal.io.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> dbf = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')

    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array([dbf.by_col('HOVAL')]).T

    Extract CRIME (crime) and INC (income) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
    independent variables passed in.

    >>> names_to_extract = ['INC', 'CRIME']
    >>> x = np.array([dbf.by_col(name) for name in names_to_extract]).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will use
    ``columbus.gal``, which contains contiguity relationships between the
    observations in the Columbus dataset we are using throughout this example.
    Note that, in order to read the file, not only to open it, we need to
    append '.read()' at the end of the command.

    >>> w = libpysal.io.open(libpysal.examples.get_path("columbus.gal"), 'r').read()

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform='r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Error(y, x, w=w, name_y='hoval', name_x=['income', 'crime'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas).

    >>> print(model.name_x)
    ['CONSTANT', 'income', 'crime', 'lambda']
    >>> np.around(model.betas, decimals=4)
    array([[47.6946],
           [ 0.7105],
           [-0.5505],
           [ 0.3257]])
    >>> np.around(model.std_err, decimals=4)
    array([12.412 ,  0.5044,  0.1785])
    >>> np.around(model.z_stat, decimals=6) #doctest: +SKIP
    array([[  3.84261100e+00,   1.22000000e-04],
           [  1.40839200e+00,   1.59015000e-01],
           [ -3.08424700e+00,   2.04100000e-03]])
    >>> round(model.sig2,4)
    198.5596

    """

    def __init__(
        self, y, x, w, slx_lags=0, slx_vars="All",vm=False, name_y=None, name_x=None, name_w=None, name_ds=None, latex=False,
            hard_bound=False):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        x_constant, name_x, warn = USER.check_constant(x, name_x)
        name_x = USER.set_name_x(name_x, x_constant)  # intialize in case of None, contains constant
        set_warn(self, warn)
        
        self.title = "GM SPATIALLY WEIGHTED LEAST SQUARES"
        if slx_lags >0:
            #lag_x = get_lags(w, x_constant[:, 1:], slx_lags)
            #x_constant = np.hstack((x_constant, lag_x))
#            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
            #name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags) # exclude constant

            x_constant,name_x = USER.flex_wx(w,x=x_constant,name_x=name_x,constant=True,
                                             slx_lags=slx_lags,slx_vars=slx_vars)

            self.title += " WITH SLX (SLX-Error)"
        
        BaseGM_Error.__init__(self, y=y, x=x_constant, w=w.sparse, hard_bound=hard_bound)
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
#        self.name_x = USER.set_name_x(name_x, x_constant)
        self.name_x = name_x  # already includes constant
        self.name_x.append("lambda")
        self.name_w = USER.set_name_w(name_w, w)
        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['x'] * (len(self.name_x) - 1) + ['lambda']
        self.output['regime'], self.output['equation'] = (0, 0)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


class BaseGM_Endog_Error(RegressionPropsY):

    """
    GMM method for a spatial error model with endogenous variables (note: no
    consistency checks, diagnostics or constant added); based on Kelejian and
    Prucha (1998, 1999) :cite:`Kelejian1998` :cite:`Kelejian1999`.

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
    z            : array
                   nxk array of variables (combination of x and yend)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations

    Examples
    --------

    >>> import libpysal
    >>> import numpy as np
    >>> import spreg
    >>> dbf = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array([dbf.by_col('CRIME')]).T
    >>> x = np.array([dbf.by_col('INC')]).T
    >>> x = np.hstack((np.ones(y.shape),x))
    >>> yend = np.array([dbf.by_col('HOVAL')]).T
    >>> q = np.array([dbf.by_col('DISCBD')]).T
    >>> w = libpysal.io.open(libpysal.examples.get_path("columbus.gal"), 'r').read()
    >>> w.transform='r'
    >>> model = spreg.error_sp.BaseGM_Endog_Error(y, x, yend, q, w=w.sparse)
    >>> np.around(model.betas, decimals=4)
    array([[82.5723],
           [ 0.581 ],
           [-1.4481],
           [ 0.3499]])

    """

    def __init__(self, y, x, yend, q, w, hard_bound=False):

        # 1a. TSLS --> \tilde{betas}
        tsls = TSLS.BaseTSLS(y=y, x=x, yend=yend, q=q)
        self.n, self.k = tsls.z.shape
        self.x = tsls.x
        self.y = tsls.y
        self.yend, self.z = tsls.yend, tsls.z

        # 1b. GMM --> \tilde{\lambda1}
        moments = _momentsGM_Error(w, tsls.u)
        lambda1 = optim_moments(moments, hard_bound=hard_bound)

        # 2a. 2SLS -->\hat{betas}
        xs = get_spFilter(w, lambda1, self.x)
        ys = get_spFilter(w, lambda1, self.y)
        yend_s = get_spFilter(w, lambda1, self.yend)
        tsls2 = TSLS.BaseTSLS(ys, xs, yend_s, h=tsls.h)

        # Output
        self.betas = np.vstack((tsls2.betas, np.array([[lambda1]])))
        self.predy = spdot(tsls.z, tsls2.betas)
        self.u = y - self.predy
        self.sig2 = float(np.dot(tsls2.u.T, tsls2.u)) / self.n
        self.e_filtered = self.u - lambda1 * w * self.u
        self.vm = self.sig2 * tsls2.varb
        self._cache = {}


class GM_Endog_Error(BaseGM_Endog_Error):

    """
    GMM method for a spatial error model with endogenous variables, with
    results and diagnostics; based on Kelejian and Prucha (1998,
    1999) :cite:`Kelejian1998` :cite:`Kelejian1999`.

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
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX-Error type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged    
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
    z            : array
                   nxk array of variables (combination of x and yend)
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

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import libpysal
    >>> import numpy as np

    Open data on Columbus neighborhood crime (49 areas) using libpysal.io.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> dbf = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),'r')

    Extract the CRIME column (crime rates) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array([dbf.by_col('CRIME')]).T

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in.

    >>> x = np.array([dbf.by_col('INC')]).T

    In this case we consider HOVAL (home value) is an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yend = np.array([dbf.by_col('HOVAL')]).T

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for HOVAL. We use DISCBD (distance to the
    CBD) for this and hence put it in the instruments parameter, 'q'.

    >>> q = np.array([dbf.by_col('DISCBD')]).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will use
    ``columbus.gal``, which contains contiguity relationships between the
    observations in the Columbus dataset we are using throughout this example.
    Note that, in order to read the file, not only to open it, we need to
    append '.read()' at the end of the command.

    >>> w = libpysal.io.open(libpysal.examples.get_path("columbus.gal"), 'r').read()

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform='r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables (exogenous and endogenous), the
    instruments and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> from spreg import GM_Endog_Error
    >>> model = GM_Endog_Error(y, x, yend, q, w=w, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    endogenous variables included.

    >>> print(model.name_z)
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> np.around(model.betas, decimals=4)
    array([[82.5723],
           [ 0.581 ],
           [-1.4481],
           [ 0.3499]])
    >>> np.around(model.std_err, decimals=4)
    array([16.1382,  1.3545,  0.7862])

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
        name_x = USER.set_name_x(name_x, x_constant) # initialize for None, includes constant
        set_warn(self, warn)
        self.title = "GM SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES"
        if slx_lags >0:
            #lag_x = get_lags(w, x_constant[:, 1:], slx_lags)
            #x_constant = np.hstack((x_constant, lag_x))
#            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
            #name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)  # exclude constant

            x_constant,name_x = USER.flex_wx(w,x=x_constant,name_x=name_x,constant=True,
                                             slx_lags=slx_lags,slx_vars=slx_vars)

            self.title += " WITH SLX (SLX-Error)"        
        BaseGM_Endog_Error.__init__(self, y=y, x=x_constant, w=w.sparse, yend=yend, q=q, hard_bound=hard_bound)
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
#        self.name_x = USER.set_name_x(name_x, x_constant)
        self.name_x = name_x  # already includes constant
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append("lambda")
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        self.output = pd.DataFrame(self.name_z,
                                   columns=['var_names'])
        self.output['var_type'] = ['x'] * len(self.name_x) + ['yend'] * len(self.name_yend) + ['lambda']
        self.output['regime'], self.output['equation'] = (0, 0)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


class GM_Combo(BaseGM_Endog_Error):

    """
    GMM method for a spatial lag and error model with endogenous variables,
    with results and diagnostics; based on Kelejian and Prucha (1998,
    1999) :cite:`Kelejian1998` :cite:`Kelejian1999`.

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
                   instruments (q)
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
    z            : array
                   nxk array of variables (combination of x and yend)
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

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import GM_Combo

    Open data on Columbus neighborhood crime (49 areas) using libpysal.io.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),'r')

    Extract the CRIME column (crime rates) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
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
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    The Combo class runs an SARAR model, that is a spatial lag+error model.
    In this case we will run a simple version of that, where we have the
    spatial effects as well as exogenous variables. Since it is a spatial
    model, we have to pass in the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> reg = GM_Combo(y, X, w=w, name_y='crime', name_x=['income'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    spatial lag of the dependent variable. We can check the betas:

    >>> print(reg.name_z)
    ['CONSTANT', 'income', 'W_crime', 'lambda']
    >>> print(np.around(np.hstack((reg.betas[:-1],np.sqrt(reg.vm.diagonal()).reshape(3,1))),3))
    [[39.059 11.86 ]
     [-1.404  0.391]
     [ 0.467  0.2  ]]

    And lambda:

    >>> print('lambda: ', np.around(reg.betas[-1], 3))
    lambda:  [-0.048]

    This class also allows the user to run a spatial lag+error model with the
    extra feature of including non-spatial endogenous regressors. This means
    that, in addition to the spatial lag and error, we consider some of the
    variables on the right-hand side of the equation as endogenous and we
    instrument for this. As an example, we will include HOVAL (home value) as
    endogenous and will instrument with DISCBD (distance to the CSB). We first
    need to read in the variables:

    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    And then we can run and explore the model analogously to the previous combo:

    >>> reg = GM_Combo(y, X, yd, q, w=w, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')
    >>> print(reg.name_z)
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']
    >>> names = np.array(reg.name_z).reshape(5,1)
    >>> print(np.hstack((names[0:4,:], np.around(np.hstack((reg.betas[:-1], np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))))
    [['CONSTANT' '50.0944' '14.3593']
     ['inc' '-0.2552' '0.5667']
     ['hoval' '-0.6885' '0.3029']
     ['W_crime' '0.4375' '0.2314']]

    >>> print('lambda: ', np.around(reg.betas[-1], 3))
    lambda:  [0.254]

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
        name_x = USER.set_name_x(name_x, x_constant)

        if slx_lags > 0:
            yend2, q2, wx = set_endog(y, x_constant[:, 1:], w, yend, q, w_lags, lag_q, slx_lags,slx_vars)
            x_constant = np.hstack((x_constant, wx))
        else:
            yend2, q2 = set_endog(y, x_constant[:, 1:], w, yend, q, w_lags, lag_q)



        set_warn(self, warn)
        # OLD
        #if slx_lags == 0:
            #yend2, q2 = set_endog(y, x_constant[:, 1:], w, yend, q, w_lags, lag_q)
        #else:
            #yend2, q2, wx = set_endog(y, x_constant[:, 1:], w, yend, q, w_lags, lag_q, slx_lags)
            #x_constant = np.hstack((x_constant, wx))

        BaseGM_Endog_Error.__init__(self, y=y, x=x_constant, w=w.sparse, yend=yend2, q=q2, hard_bound=hard_bound)

        self.rho = self.betas[-2]
        self.predy_e, self.e_pred, warn = sp_att(
            w, self.y, self.predy, yend2[:, -1].reshape(self.n, 1), self.rho
        )
        set_warn(self, warn)
        self.title = "SPATIALLY WEIGHTED 2SLS - GM-COMBO MODEL"
        # OLD
        #if slx_lags > 0:
#            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
            #name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)   # exclude constant
            #self.title += " WITH SLX (GNSM)"

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
            self.title += " WITH SLX (GNSM)"

        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
#        self.name_x = USER.set_name_x(name_x, x_constant)
        self.name_x = name_x  # constant already in list
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))


        self.name_z = self.name_x + self.name_yend
        self.name_z.append("lambda")
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
        

        #self.output['var_type'] = ['x'] * len(self.name_x) + ['yend'] * (len(self.name_yend) - 1) + ['rho', 'lambda']
        self.output['var_type'] = var_types

        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top = _spat_pseudo_r2(self)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


class GMM_Error(GM_Error, GM_Endog_Error, GM_Combo, GM_Error_Het, GM_Endog_Error_Het, 
               GM_Combo_Het, GM_Error_Hom, GM_Endog_Error_Hom, GM_Combo_Hom):

    """
    Wrapper function to call any of the GMM methods for a spatial error model available in spreg

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object (always needed)
    yend         : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   endogenous variable (if any)
    q            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (if any)
                   (note: this should not contain any variables from x)
    estimator    : string
                   Choice of estimator to be used. Options are: 'het', which
                   is robust to heteroskedasticity, 'hom', which assumes
                   homoskedasticity, and 'kp98', which does not provide
                   inference on the spatial parameter for the error term.
    add_wy       : boolean
                   If True, then a spatial lag of the dependent variable is included.           
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX-Error or GNSM type. 
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged            
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output                   
    name_ds      : string
                   Name of dataset for use in output
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    spat_diag    : boolean, ignored, included for compatibility with other models
    **kwargs     : keywords
                   Additional arguments to pass on to the estimators. 
                   See the specific functions for details on what can be used.

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
    name_yend    : list of strings (optional)
                    Names of endogenous variables for use in output
    name_z       : list of strings (optional)
                    Names of exogenous and endogenous variables for use in
                    output
    name_q       : list of strings (optional)
                    Names of external instruments
    name_h       : list of strings (optional)
                    Names of all instruments used in ouput                   
    

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``libpysal`` to
    handle the weights and file management.

    >>> import numpy as np
    >>> import libpysal
    >>> from libpysal.examples import load_example

    Open data on NCOVR US County Homicides (3085 areas) using libpysal.io.open().
    This is the DBF associated with the NAT shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> nat = load_example('Natregimes')
    >>> db = libpysal.io.open(nat.get_path("natregimes.dbf"),'r')

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

     Since we want to run a spatial error model, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations. To do that, we can open an already existing gal file or
    create a new one. In this case, we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Rook.from_shapefile(nat.get_path("natregimes.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    The GMM_Error class can run error models and SARAR models, that is a spatial lag+error model.
    In this example we will run a simple version of the latter, where we have the
    spatial effects as well as exogenous variables. Since it is a spatial
    model, we have to pass in the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> from spreg import GMM_Error
    >>> model = GMM_Error(y, x, w=w, add_wy=True, name_y=y_var, name_x=x_var, name_ds='NAT')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them.

    >>> print(model.output)
      var_names coefficients   std_err    zt_stat      prob
    0  CONSTANT     2.176007  1.115807   1.950165  0.051156
    1      PS90     1.108054  0.207964   5.328096       0.0
    2      UE90     0.664362  0.061294   10.83893       0.0
    3    W_HR90    -0.066539  0.154395  -0.430964  0.666494
    4    lambda     0.765087   0.04268  17.926245       0.0

    This class also allows the user to run a spatial lag+error model with the
    extra feature of including non-spatial endogenous regressors. This means
    that, in addition to the spatial lag and error, we consider some of the
    variables on the right-hand side of the equation as endogenous and we
    instrument for this. In this case we consider RD90 (resource deprivation)
    as an endogenous regressor.  We use FP89 (families below poverty)
    for this and hence put it in the instruments parameter, 'q'.

    >>> yd_var = ['RD90']
    >>> yd = np.array([db.by_col(name) for name in yd_var]).T
    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    And then we can run and explore the model analogously to the previous combo:

    >>> model = GMM_Error(y, x, yend=yd, q=q, w=w, add_wy=True, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_ds='NAT')
    >>> print(model.output)
      var_names coefficients   std_err    zt_stat      prob
    0  CONSTANT      5.44035  0.560476   9.706652       0.0
    1      PS90     1.427042    0.1821   7.836572       0.0
    2      UE90    -0.075224  0.050031  -1.503544  0.132699
    3      RD90     3.316266  0.261269  12.692924       0.0
    4    W_HR90     0.200314  0.057433   3.487777  0.000487
    5    lambda     0.136933  0.070098   1.953457  0.050765

    The class also allows for estimating a GNS model by adding spatial lags of the exogenous variables, using the argument slx_lags:

    >>> model = GMM_Error(y, x, w=w, add_wy=True, slx_lags=1, name_y=y_var, name_x=x_var, name_ds='NAT')
    >>> print(model.output)
      var_names coefficients   std_err   zt_stat      prob
    0  CONSTANT    -0.554756  0.551765  -1.00542  0.314695
    1      PS90      1.09369  0.225895  4.841583  0.000001
    2      UE90     0.697393  0.082744  8.428291       0.0
    3    W_PS90    -1.533378  0.396651 -3.865811  0.000111
    4    W_UE90    -1.107944   0.33523 -3.305028   0.00095
    5    W_HR90     1.529277  0.389354  3.927728  0.000086
    6    lambda    -0.917928  0.079569 -11.53625       0.0


    """

    def __init__(
        self, y, x, w, yend=None, q=None, estimator='het', add_wy=False, slx_lags=0, slx_vars="All",vm=False, name_y=None, name_x=None, name_w=None, name_yend=None,
        name_q=None, name_ds=None, latex=False, hard_bound=False,spat_diag=False, **kwargs):

        if estimator == 'het':
            if yend is None and not add_wy:
                GM_Error_Het.__init__(self, y=y, x=x, w=w, slx_lags=slx_lags, slx_vars=slx_vars, vm=vm, name_y=name_y, name_x=name_x, 
                                      name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound, **kwargs)
            elif yend is not None and not add_wy:
                GM_Endog_Error_Het.__init__(self, y=y, x=x, yend=yend, q=q, w=w, slx_lags=slx_lags, slx_vars=slx_vars, vm=vm, name_y=name_y, name_x=name_x,
                                            name_yend=name_yend, name_q=name_q, name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound, **kwargs)
            elif add_wy:
                GM_Combo_Het.__init__(self, y=y, x=x, yend=yend, q=q, w=w, slx_lags=slx_lags, slx_vars=slx_vars, vm=vm, name_y=name_y, name_x=name_x,
                                            name_yend=name_yend, name_q=name_q, name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound, **kwargs)
            else:
                set_warn(self, 'Combination of arguments passed to GMM_Error not allowed. Using default arguments instead.')
                GM_Error_Het.__init__(self, y=y, x=x, w=w, slx_lags=slx_lags, slx_vars=slx_vars, vm=vm, name_y=name_y, name_x=name_x, 
                                      name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound)
        elif estimator == 'hom':
            if yend is None and not add_wy:
                GM_Error_Hom.__init__(self, y=y, x=x, w=w, slx_lags=slx_lags, slx_vars=slx_vars,vm=vm, name_y=name_y, name_x=name_x, 
                                      name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound, **kwargs)
            elif yend is not None and not add_wy:
                GM_Endog_Error_Hom.__init__(self, y=y, x=x, yend=yend, q=q, w=w, slx_lags=slx_lags, slx_vars=slx_vars, vm=vm, name_y=name_y, name_x=name_x,
                                            name_yend=name_yend, name_q=name_q, name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound, **kwargs)
            elif add_wy:
                GM_Combo_Hom.__init__(self, y=y, x=x, yend=yend, q=q, w=w, slx_lags=slx_lags, slx_vars=slx_vars, vm=vm, name_y=name_y, name_x=name_x,
                                            name_yend=name_yend, name_q=name_q, name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound, **kwargs)
            else:
                set_warn(self, 'Combination of arguments passed to GMM_Error not allowed. Using default arguments instead.')
                GM_Error_Hom.__init__(self, y=y, x=x, w=w, slx_lags=slx_lags, slx_vars=slx_vars,vm=vm, name_y=name_y, name_x=name_x, 
                                      name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound)
        elif estimator == 'kp98':
            if yend is None and not add_wy:
                GM_Error.__init__(self, y=y, x=x, w=w, slx_lags=slx_lags, slx_vars=slx_vars, vm=vm, name_y=name_y, name_x=name_x, 
                                      name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound, **kwargs)
            elif yend is not None and not add_wy:
                GM_Endog_Error.__init__(self, y=y, x=x, yend=yend, q=q, w=w, slx_lags=slx_lags, slx_vars=slx_vars, vm=vm, name_y=name_y, name_x=name_x,
                                            name_yend=name_yend, name_q=name_q, name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound, **kwargs)
            elif add_wy:
                GM_Combo.__init__(self, y=y, x=x, yend=yend, q=q, w=w, slx_lags=slx_lags, slx_vars=slx_vars, vm=vm, name_y=name_y, name_x=name_x,
                                            name_yend=name_yend, name_q=name_q, name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound, **kwargs)
            else:
                set_warn(self, 'Combination of arguments passed to GMM_Error not allowed. Using default arguments instead.')
                GM_Error.__init__(self, y=y, x=x, w=w, slx_lags=slx_lags, slx_vars=slx_vars, vm=vm, name_y=name_y, name_x=name_x, 
                                      name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound)
        else:
            set_warn(self, 'Combination of arguments passed to GMM_Error not allowed. Using default arguments instead.')
            GM_Error_Het.__init__(self, y=y, x=x, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x, 
                                    name_w=name_w, name_ds=name_ds, latex=latex, hard_bound=hard_bound)


def _momentsGM_Error(w, u):
    try:
        wsparse = w.sparse
    except:
        wsparse = w
    n = wsparse.shape[0]
    u2 = np.dot(u.T, u)
    wu = wsparse * u
    uwu = np.dot(u.T, wu)
    wu2 = np.dot(wu.T, wu)
    wwu = wsparse * wu
    uwwu = np.dot(u.T, wwu)
    wwu2 = np.dot(wwu.T, wwu)
    wuwwu = np.dot(wu.T, wwu)
    wtw = wsparse.T * wsparse
    trWtW = np.sum(wtw.diagonal())
    g = np.array([[u2[0][0], wu2[0][0], uwu[0][0]]]).T / n
    G = (
        np.array(
            [
                [2 * uwu[0][0], -wu2[0][0], n],
                [2 * wuwwu[0][0], -wwu2[0][0], trWtW],
                [uwwu[0][0] + wu2[0][0], -wuwwu[0][0], 0.0],
            ]
        )
        / n
    )
    return [G, g]


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
    #reg = GM_Error(y, X, w=w, name_x=['inc'], name_y='hoval', name_ds='columbus', vm=True)
    #reg = GM_Endog_Error(y, X, yd, q, w=w, name_x=['inc'], name_y='hoval', name_yend=['crime'],
    #                         name_q=['discbd'], name_ds='columbus',vm=True)
    reg = GM_Combo(y, X, yd, q, w=w, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'],
                       name_ds='columbus', vm=True)

    print(reg.output)
    print(reg.summary)