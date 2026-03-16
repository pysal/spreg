import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.stats as stats
import scipy.sparse as sps
from scipy.optimize import minimize_scalar, minimize
from libpysal.weights import WSP
from .error_sp_het import BaseGM_Error_Het
from .ml_error import BaseML_Error
from .diagnostics_panel import BSK_tests
from .utils import RegressionPropsY, RegressionPropsVM, set_warn, optim_moments, get_spFilter, spdot, inverse_prod
from .sputils import spdot, spfill_diagonal, spinv
from .ols import BaseOLS
from .panel_utils import prepare_panel, demean_panel
from .output import output, _nonspat_mid, _nonspat_top, _summary_iteration, _summary_impacts
from . import regimes as REGI
from . import user_output as USER
from . import diagnostics as DIAG

__author__ = "Luc Anselin lanselin@gmail.com, Pedro Amaral pedrovma@gmail.com, Pablo Estrada pabloestradace@gmail.com"

__all__ = ["PooledOLS", "PanelFE", "PanelRE", "GM_ErrorPooled", "ML_ErrorPooled", 
           "GM_ErrorRE", "ML_ErrorFE", "ML_ErrorRE", "ML_LagFE", "ML_LagRE"]

class PooledOLS(BaseOLS, RegressionPropsY, RegressionPropsVM):
    """
    Pooled OLS for Panel Data.
    
    This is the baseline model for panel data analysis. It stacks the 
    cross-sectional data over time and runs a standard OLS regression, 
    ignoring specific individual or time effects (unless dummies are manually provided).

    Parameters
    ----------
    y            : numpy.ndarray or pandas object
                   nxt or (nxt)x1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, no constant
    w            : pysal W object, optional
                   Spatial weights object. Not required for estimation, but
                   required for spatial diagnostic tests (LM tests).
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
    robust       : string, optional
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given. 
                   If 'hac', then a HAC consistent estimator is given.
    vm           : boolean
                   if True, include variance-covariance matrix in summary
                   results
    nonspat_diag : boolean
                   if True, include non-spatial diagnostic tests in summary (default: False)
    spat_diag    : boolean
                   If True, then compute the BSK tests.
    BSK_list    : list of strings
                     List of BSK tests to compute if spat_diag is True. 
                     Options are "all", "LMJ", "LM1", "LM2", "LMC_spatial", "LMC_RE" 
                     (default: ["LMJ", "LM1", "LM2"])
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
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations (N * T)
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   Log-likelihood of the estimation
    aic          : float
                   Akaike information criterion
    schwarz      : float
                   Schwarz criterion

    Examples
    --------

    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spreg import PooledOLS
    
    In this example, we will use data on NCOVR US County Homicides (3085 areas) included in libpysal.

    >>> libpysal.examples.load_example('NCOVR')
    >>> db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    >>> name_ds = "NCOVR.shp"

    Extract the HR (homicide rates) data in the 70's, 80's and 90's from the dataframe
    and make it the dependent variable for the regression. Note that the data can also
    be passed in the long format instead of wide format (i.e. a vector with n*t rows
    and a single column for the dependent variable and a matrix of dimension n*txk
    for the independent variables).

    >>> y_name = ['HR70','HR80','HR90']
    >>> y = db[y_name]

    Extract RD and PS in the same time periods from the dataframe to be used as
    independent variables in the regression. 
    IMPORTANT: The data must be organized in a way that all time periods of a 
    given variable are side-by-side and in the correct time order.
    That is: x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    By default a vector of ones will be added to the independent variables passed in.

    >>> x_names = ['RD70','RD80','RD90','PS70','PS80','PS90']
    >>> x = db[x_names]

    Even if we are not running a spatial model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model for the diagnostics. To do that, 
    we can open an already existing gal file or create a new one. In this case, 
    we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = 'r'
    >>> name_w = "NCOVR.gal"

    We are all set with the preliminaries, we are good to run the model.

    >>> reg = PooledOLS(y, x, w)

    Once we have run the model, we can explore a little bit the output with 
    the command print(reg.summary).                   
    """

    def __init__(
        self,
        y,
        x,
        w,
        slx_lags = 0,
        slx_vars = "all",
        robust=None,
        vm=False,
        nonspat_diag=False,
        spat_diag=True,        
        BSK_list=["LMJ", "LM1", "LM2"],
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
    ):

        self.title = "POOLED OLS PANEL"
        bigy, bigx, self.name_y, self.name_x, w, warn, self.slx_lags, self.title, kx, kwx, self.t = prepare_panel(
            y, x, w, name_y, name_x, slx_lags, slx_vars, self.title, add_constant=True)
        any(set_warn(self, i) for i in warn)
        
        self.k = bigx.shape[1]
        
        BaseOLS.__init__(self, bigy, bigx, robust=robust, sig2n_k=True)

        self.name_ds = USER.set_name_ds(name_ds)
        self.name_w = USER.set_name_w(name_w, w)
        self.robust = USER.set_robust(robust)

        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['o'] + ['x'] * (kx-1) + ['wx'] * kwx
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top, self.other_mid, other_end = ("", "", "") 

        self.other_top += _nonspat_top(self)
        if nonspat_diag:
            self.other_mid += _nonspat_mid(self)
        if spat_diag:
            self.bsk = BSK_tests(self, w, which=BSK_list)
            other_end += "\nSPATIAL DIAGNOSTIC TESTS\n"
            other_end += f"{'-' * 84 }\n"
            other_end += "TEST                             DF        VALUE           PROB\n"
            for i in range(len(self.bsk)):
                other_end += f"{self.bsk['Test'][i]:30s} {self.bsk['df'][i]:3d}   {self.bsk['Statistic'][i]:12.4f}       {self.bsk['p-value'][i]:8.5f}\n"  
        output(reg=self, vm=vm, robust=robust, other_end=other_end, latex=latex)

class PanelFE(RegressionPropsY, RegressionPropsVM):
    """
    Fixed Effects (One-Way) OLS for Panel Data.
    
    This model controls for time-invariant unobserved heterogeneity by 
    demeaning the data (Within Transformation) before estimation. 
    Time-invariant variables (like Region dummies) will be dropped 
    due to demeaning.

    Parameters
    ----------
    y            : numpy.ndarray or pandas object
                   nxt or (nxt)x1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables.
    w            : pysal W object
                   Spatial weights object. Required to determine cross-sectional
                   dimension (N) and for spatial diagnostics.
    slx_lags     : integer
                    Number of spatial lags of X to include in the model specification.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                    to be lagged
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
    latex       : boolean
                    Specifies if summary is to be printed in latex format

    Attributes
    ----------
    output       : dataframe
                   regression results pandas dataframe
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients (within estimator)
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    fixed_effects: array
                   nx1 array of the estimated fixed effects (mu_i)
    n            : integer
                   Total number of observations (N * T)
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    logll        : float
                   Log-likelihood of the estimation
    mean_mu_i     : float
                     Mean of the individual effects (mu_i)   

    Examples
    --------

    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spreg import PanelFE
    
    In this example, we will use data on NCOVR US County Homicides (3085 areas) included in libpysal.

    >>> libpysal.examples.load_example('NCOVR')
    >>> db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    >>> name_ds = "NCOVR.shp"

    Extract the HR (homicide rates) data in the 70's, 80's and 90's from the dataframe
    and make it the dependent variable for the regression. Note that the data can also
    be passed in the long format instead of wide format (i.e. a vector with n*t rows
    and a single column for the dependent variable and a matrix of dimension n*txk
    for the independent variables).

    >>> y_name = ['HR70','HR80','HR90']
    >>> y = db[y_name]

    Extract RD and PS in the same time periods from the dataframe to be used as
    independent variables in the regression. 
    IMPORTANT: The data must be organized in a way that all time periods of a 
    given variable are side-by-side and in the correct time order.
    That is: x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    By default a vector of ones will be added to the independent variables passed in.

    >>> x_names = ['RD70','RD80','RD90','PS70','PS80','PS90']
    >>> x = db[x_names]

    Even if we are not running a spatial model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    for the diagnostics. To do that, 
    we can open an already existing gal file or create a new one. In this case, 
    we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = 'r'
    >>> name_w = "NCOVR.gal"

    We are all set with the preliminaries, we are good to run the model.

    >>> reg = PanelFE(y, x, w)

    Once we have run the model, we can explore a little bit the output with 
    the command print(reg.summary).  
    """

    def __init__(
        self,
        y,
        x,
        w,
        slx_lags=0,
        slx_vars="all",
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
    ):
        
        self.title = "FIXED EFFECTS PANEL (ONE-WAY)"
        bigy, bigx, self.name_y, self.name_x, w, warn, self.slx_lags, self.title, kx, kwx, self.t = prepare_panel(
            y, x, w, name_y, name_x, slx_lags, slx_vars, self.title, add_constant=False)
        any(set_warn(self, i) for i in warn)
  
        self.k = bigx.shape[1]

        # Demean
        N = w.n
        NT = bigy.shape[0]
        T = NT // N
        
        y_mat = bigy.reshape((T, N))
        x_mat = bigx.reshape((T, N, self.k))
        
        y_mean = y_mat.mean(axis=0)
        x_mean = x_mat.mean(axis=0)
        
        dem_y = bigy - np.tile(y_mean, T).reshape(-1, 1)
        
        dem_x = np.zeros_like(bigx)
        for i in range(self.k):
            col_mean = x_mean[:, i] 
            col_vals = bigx[:, i] 
            dem_x[:, i] = col_vals - np.tile(col_mean, T)

        reg_ols = BaseOLS(dem_y, dem_x, sig2n_k=False)
        self.y, self.x = bigy, bigx
        self.betas = reg_ols.betas
        self.predy = np.dot(bigx, self.betas)
        self.u = bigy - self.predy
        self.xtxi = la.inv(np.dot(bigx.T, bigx))
        self.df_resid = NT - N

        ssr = reg_ols.u.T @ reg_ols.u
        self.sig2 = ssr[0, 0] / self.df_resid
        self.vm = np.dot(self.sig2, reg_ols.xtxi)
        self.std_err = np.sqrt(np.diag(self.vm))
        self.fixed_effects = y_mean.reshape(-1, 1) - (x_mean @ self.betas)

        self.name_ds = USER.set_name_ds(name_ds)
        self.name_w = USER.set_name_w(name_w, w)
        self.t, self.n = T, NT

        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['x'] * kx + ['wx'] * kwx
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top, self.other_mid, other_end = ("", "", "") 
        self.other_top += _nonspat_top(self)

        self.mu_i = self.u.reshape(self.t, N).mean(axis=0).reshape(-1, 1)        
        self.mean_mu_i = self.mu_i.mean()
        self.other_top += "%-20s:%12.4f\n" % (
            "Fixed-effects mean", self.mean_mu_i)        

        output(reg=self, vm=vm, other_end=other_end, latex=latex)

class PanelRE(RegressionPropsY, RegressionPropsVM):
    """
    Random Effects (One-Way) GLS for Panel Data.
    
    Includes the Swamy-Arora estimator for variance components and 
    the classic Hausman Specification Test (not robust to spatial autocorrelation).

    Parameters
    ----------
    y            : numpy.ndarray or pandas object
                   nxt or (nxt)x1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables.
    w            : pysal W object
                   Spatial weights object. Required to determine cross-sectional
                   dimension (N).
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
    spat_diag    : boolean
                   If True (default), then compute the 'LMC_spatial' BSK test.
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
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)    
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations (N * T)
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables
    theta        : float
                   The quasi-demeaning weight (0 = Pooled, 1 = FE)
    hausman_stat : float
                   Chi-square statistic for the Hausman test
    hausman_p    : float
                   P-value for the Hausman test
    sigma2_mu     : float
                   Estimated variance of the random effects
    sigma2_epsilon : float
                   Estimated variance of the idiosyncratic errors

    Examples
    --------

    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spreg import PanelRE
    
    In this example, we will use data on NCOVR US County Homicides (3085 areas) included in libpysal.

    >>> libpysal.examples.load_example('NCOVR')
    >>> db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    >>> name_ds = "NCOVR.shp"

    Extract the HR (homicide rates) data in the 70's, 80's and 90's from the dataframe
    and make it the dependent variable for the regression. Note that the data can also
    be passed in the long format instead of wide format (i.e. a vector with n*t rows
    and a single column for the dependent variable and a matrix of dimension n*txk
    for the independent variables).

    >>> y_name = ['HR70','HR80','HR90']
    >>> y = db[y_name]

    Extract RD and PS in the same time periods from the dataframe to be used as
    independent variables in the regression. 
    IMPORTANT: The data must be organized in a way that all time periods of a 
    given variable are side-by-side and in the correct time order.
    That is: x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    By default a vector of ones will be added to the independent variables passed in.

    >>> x_names = ['RD70','RD80','RD90','PS70','PS80','PS90']
    >>> x = db[x_names]

    Even if we are not running a spatial model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    for the diagnostics. To do that, 
    we can open an already existing gal file or create a new one. In this case, 
    we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = 'r'
    >>> name_w = "NCOVR.gal"

    We are all set with the preliminaries, we are good to run the model.

    >>> reg = PanelRE(y, x, w)

    Once we have run the model, we can explore a little bit the output with 
    the command print(reg.summary).  
    """

    def __init__(
        self,
        y,
        x,
        w,
        slx_lags=0,
        slx_vars="all",
        spat_diag=True,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
    ):
        
        self.title = "RANDOM EFFECTS (ONE-WAY) PANEL"
        bigy, bigx, self.name_y, self.name_x, w, warn, self.slx_lags, self.title, kx, kwx, self.t = prepare_panel(
            y, x, w, name_y, name_x, slx_lags, slx_vars, self.title, add_constant=True)
        any(set_warn(self, i) for i in warn)

        self.k = bigx.shape[1]
        
        N = w.n
        T = bigy.shape[0] // N
        
        y_mat = bigy.reshape((T, N))
        x_mat = bigx.reshape((T, N, self.k))
        y_mean = y_mat.mean(axis=0)
        x_mean = x_mat.mean(axis=0)

        y_mean_long = np.tile(y_mean, T).reshape(-1, 1)
        x_mean_long = np.zeros_like(bigx)
        for i in range(self.k):
            x_mean_long[:, i] = np.tile(x_mean[:, i], T)

        # Within Estimation
        y_dem = bigy - y_mean_long
        x_dem = bigx - x_mean_long
        
        var_check = np.var(x_dem, axis=0)
        valid_cols = var_check > 1e-12 
        
        beta_fe = None
        xtx_inv_fe = None
        
        if np.sum(valid_cols) > 0:
            x_dem_slim = x_dem[:, valid_cols]
            xtx = x_dem_slim.T @ x_dem_slim
            xtx_inv_fe = np.linalg.inv(xtx)
            xty = x_dem_slim.T @ y_dem
            beta_fe = xtx_inv_fe @ xty
            e_fe = y_dem - (x_dem_slim @ beta_fe)
        else:
            e_fe = y_dem

        df_fe = (N * T) - N - np.sum(valid_cols)
        self.sigma2_epsilon = (e_fe.T @ e_fe)[0, 0] / df_fe

        # Between Estimation
        y_mean_col = y_mean.reshape(-1, 1)
        xtx_b = x_mean.T @ x_mean
        xty_b = x_mean.T @ y_mean_col
        
        try:
            beta_b = np.linalg.solve(xtx_b, xty_b)
        except np.linalg.LinAlgError:
            beta_b = np.linalg.pinv(xtx_b) @ xty_b
            
        e_b = y_mean_col - (x_mean @ beta_b)
        ssr_b = (e_b.T @ e_b)[0, 0]
        sigma2_b = ssr_b / (N - self.k)
        
        self.sigma2_mu = sigma2_b - (self.sigma2_epsilon / T)
        
        if self.sigma2_mu < 0:
            self.sigma2_mu = 0.0
            set_warn(self, "WARNING: Random effects variance is negative (Sigma2_mu=0).")

        denom = self.sigma2_epsilon + T * self.sigma2_mu
        self.theta = 1 - np.sqrt(self.sigma2_epsilon / denom)

        y_star = bigy - (self.theta * y_mean_long)
        x_star = bigx - (self.theta * x_mean_long)

        reg_ols = BaseOLS(y_star, x_star, sig2n_k=False)
        self.y, self.x = bigy, bigx 
        self.betas = reg_ols.betas
        self.predy = np.dot(bigx, self.betas)
        self.u = bigy - self.predy
        self.n = bigy.shape[0]
        self.sig2 = reg_ols.sig2
        self.sig2n = self.sig2
        self.vm = np.dot(self.sig2, reg_ols.xtxi)
        self.std_err = np.sqrt(np.diag(self.vm))

        # Hausman Test
        self.hausman_df = np.sum(valid_cols)
        if beta_fe is not None and self.hausman_df > 0:
            beta_re_sub = self.betas[valid_cols]
            var_fe = self.sigma2_epsilon * xtx_inv_fe
            var_re_sub = self.vm[np.ix_(valid_cols, valid_cols)]
            
            b_diff = beta_fe - beta_re_sub
            v_diff = var_fe - var_re_sub
            
            try:
                h_stat = b_diff.T @ np.linalg.inv(v_diff) @ b_diff
                self.hausman_stat = h_stat[0, 0]
                self.hausman_p = 1 - stats.chi2.cdf(self.hausman_stat, df=self.hausman_df)
            except np.linalg.LinAlgError:
                self.hausman_stat = np.nan
                self.hausman_p = np.nan
                set_warn(self, "Hausman test failed (matrix inversion error).")
        else:
            self.hausman_stat = np.nan
            self.hausman_p = np.nan

        # Output 
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_w = USER.set_name_w(name_w, w)
        
        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['o'] + ['x'] * (kx-1) + ['wx'] * kwx
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top, self.other_mid, other_end = ("", "", "") 
        self.other_top += _nonspat_top(self)
                
        other_end += "\n"
        other_end += "VARIANCE COMPONENTS ESTIMATES\n"
        other_end += f"{"-" * 84 }\n"
        other_end += f"Sigma2_epsilon (Idiosyncratic)    {self.sigma2_epsilon:12.4f}\n"
        other_end += f"Sigma2_mu (Individual)            {self.sigma2_mu:12.4f}\n"
        other_end += f"Theta (Quasi-demeaning weight)    {self.theta:12.4f}\n"
        other_end += f"{"-" * 84 }\n"

        if spat_diag or not np.isnan(self.hausman_stat):
            other_end += "\nDIAGNOSTIC TESTS\n"
            other_end += f"{"-" * 84 }\n"
            other_end += "TEST                             DF        VALUE           PROB\n"
            
            if not np.isnan(self.hausman_stat):
                other_end += f"Hausman Specification Test     {self.hausman_df:3d}   {self.hausman_stat:12.4f}       {self.hausman_p:8.5f}\n"
        
            if spat_diag:
                self.bsk = BSK_tests(self, w, which=['LMC_spatial'])
                for i in range(len(self.bsk)):
                    other_end += f"{self.bsk['Test'][i]:30s} {self.bsk['df'][i]:3d}   {self.bsk['Statistic'][i]:12.4f}       {self.bsk['p-value'][i]:8.5f}\n"  
        output(reg=self, vm=vm, other_end=other_end, latex=latex)        

class BaseGM_ErrorPooled(BaseGM_Error_Het):
    """
    Base computation class for Pooled Spatial Error Model (SEM) via GMM.
    No data checks or formatting.

    Parameters
    ----------
    y            : array
                    (NT x 1) array for dependent variable
    x            : array
                Two dimensional array with NT rows and one column for each
                independent (exogenous) variable, including the constant
    w            : PySAL (N x N) weights matrix
    max_iter     : int
                Maximum number of iterations of steps 2a and 2b from
                :cite:`Arraiz2010`. Note: epsilon provides an additional
                stop condition.
    step1c       : boolean
                If True, then include Step 1c from :cite:`Arraiz2010`.

    """
    def __init__(self, y, x, w, max_iter=1, step1c=False):
        N = w.n
        T = y.shape[0] // N
        
        W_sparse = w.sparse
        I_T = sps.eye(T)
        W_full = sps.kron(I_T, W_sparse, format='csr')
        
        BaseGM_Error_Het.__init__(self, y, x, W_full, max_iter=max_iter, step1c=step1c)

        self.w = w

class GM_ErrorPooled(BaseGM_ErrorPooled):
    """
    Pooled Spatial Error Model (SEM) for Panel Data via GMM.

    Parameters
    ----------
    y          : array or pandas DataFrame
                 n*tx1 or nxt array for dependent variable
    x          : array or pandas DataFrame
                 Two dimensional array or DF with n*t rows and k columns for
                 independent (exogenous) variable or n rows and k*t columns
                 (note, must not include a constant term)
    w          : spatial weights object
                 Spatial weights matrix, nxn
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
    nonspat_diag  : boolean
                   If True (default), then compute the 'LMC_RE' BSK test.
    maxit         : integer
                   Maximum number of iterations of steps 2a and 2b from
    step1c       : boolean
                   If True, then include Step 1c from :cite:`Arraiz2010                   
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string or list of strings
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
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)    
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    vm           : array
                   Variance covariance matrix (kxk)
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output                   

    Examples
    --------

    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spreg import GM_ErrorPooled
    
    In this example, we will use data on NCOVR US County Homicides (3085 areas) included in libpysal.

    >>> libpysal.examples.load_example('NCOVR')
    >>> db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    >>> name_ds = "NCOVR.shp"

    Extract the HR (homicide rates) data in the 70's, 80's and 90's from the dataframe
    and make it the dependent variable for the regression. Note that the data can also
    be passed in the long format instead of wide format (i.e. a vector with n*t rows
    and a single column for the dependent variable and a matrix of dimension n*txk
    for the independent variables).

    >>> y_name = ['HR70','HR80','HR90']
    >>> y = db[y_name]

    Extract RD and PS in the same time periods from the dataframe to be used as
    independent variables in the regression. 
    IMPORTANT: The data must be organized in a way that all time periods of a 
    given variable are side-by-side and in the correct time order.
    That is: x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    By default a vector of ones will be added to the independent variables passed in.

    >>> x_names = ['RD70','RD80','RD90','PS70','PS80','PS90']
    >>> x = db[x_names]

    Since we are running a spatial model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations. To do that, 
    we can open an already existing gal file or create a new one. In this case, 
    we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = 'r'
    >>> name_w = "NCOVR.gal"

    We are all set with the preliminaries, we are good to run the model.

    >>> reg = GM_ErrorPooled(y, x, w)

    Once we have run the model, we can explore a little bit the output with 
    the command print(reg.summary).      
    """

    def __init__(
        self,
        y,
        x,
        w,
        slx_lags=0,
        slx_vars="all",
        nonspat_diag=True,
        max_iter=1,
        step1c=False,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
    ):

        self.title = "GMM POOLED SPATIAL ERROR MODEL (SEM)"
        bigy, bigx, self.name_y, self.name_x, w, warn, self.slx_lags, self.title, kx, kwx, self.t = prepare_panel(
            y, x, w, name_y, name_x, slx_lags, slx_vars, self.title, add_constant=True)
        any(set_warn(self, i) for i in warn)
        self.k = bigx.shape[1]
        
        BaseGM_ErrorPooled.__init__(self, bigy, bigx, w, max_iter=max_iter, step1c=step1c)
        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_x.append("lambda")

        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['o'] + ['x'] * (kx-1) + ['wx'] * kwx + ['lambda']
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top = _summary_iteration(self)

        if nonspat_diag:
            self.bsk = BSK_tests(self, w, which=['LMC_RE'])
            other_end = "\nDIAGNOSTIC TESTS\n"
            other_end += f"{"-" * 84 }\n"
            other_end += "TEST                             DF        VALUE           PROB\n"
            for i in range(len(self.bsk)):
                other_end += f"{self.bsk['Test'][i]:30s} {self.bsk['df'][i]:3d}   {self.bsk['Statistic'][i]:12.4f}       {self.bsk['p-value'][i]:8.5f}\n"  
            other_end += "Warning: The properties of the LM Conditional RE test using GM have not been\n         established. We recommend using ML_ErrorPooled.\n"

        output(reg=self, vm=vm, robust=False, other_end=other_end, latex=latex)

class BaseML_ErrorPooled(BaseML_Error):
    """
    Base computation class for Pooled Spatial Error Model (SEM) via ML.
    No data checks or formatting.

    Parameters
    ----------
    y            : array
                (NT x 1) array for dependent variable
    x            : array
                Two dimensional array with NT rows and one column for each
                independent (exogenous) variable, including the constant
    w            : PySAL (N x N) weights matrix
    method       : string
                   if 'full', brute force calculation (full matrix expressions)
                   if 'ord', Ord eigenvalue calculation
                   if 'LU', LU decomposition for sparse matrices
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and inverse_product

    """
    def __init__(self, y, x, w, method='LU', epsilon=0.0000001):
        N = w.n    
        T = y.shape[0] // N
        
        w_sparse = w.sparse
        I_T = sps.eye(T)
        W_full_sparse = sps.kron(I_T, w_sparse, format='csc')
        w_full_wsp = WSP(W_full_sparse)
        
        if method != 'LU':
            print("Warning: Pooled ML estimation enforces method='LU' for efficiency.")
            method = 'LU'
            
        BaseML_Error.__init__(self, y, x, w_full_wsp, method=method, epsilon=epsilon)

        self.w = w       


class ML_ErrorPooled(BaseML_ErrorPooled):
    """
    Pooled Spatial Error Model (SEM) for Panel Data via ML.
    
    Parameters
    ----------
    y            : array
                   (NT x 1) array for dependent variable
    x            : array
                   Two dimensional array with NT rows and one column for each
                   independent (exogenous) variable, including the constant
    w            : PySAL W object
                   Spatial weights matrix (N x N)
    method       : string
                   if 'LU', LU decomposition for sparse matrices (Recommended for Panel)
                   if 'ord', Ord eigenvalue calculation (Slow for large Panel)
                   if 'full', brute force (Do not use for Panel)
    epsilon      : float
                   tolerance criterion in minimize_scalar function and inverse_product
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   Default is 0.
    slx_vars     : string
                   List of variables to apply spatial lag to. Default is "all".
    nonspat_diag : boolean
                   If True (default), then compute the 'LMC_RE' BSK test.
    vm           : boolean
                   If True, include variance-covariance matrix in summary results.
    name_y       : string
                   Name of dependent variable for output.
    name_x       : list of strings
                   Names of independent variables for output.
    name_w       : string
                   Name of weights matrix for output.
    name_ds      : string
                   Name of dataset for output.
    latex        : boolean
                   If True, output summary tables in LaTeX format.

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
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations (N * T)
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   Log-likelihood of the estimation
    aic          : float
                   Akaike information criterion
    schwarz      : float
                   Schwarz criterion

    Examples
    --------

    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spreg import ML_ErrorPooled
    
    In this example, we will use data on NCOVR US County Homicides (3085 areas) included in libpysal.

    >>> libpysal.examples.load_example('NCOVR')
    >>> db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    >>> name_ds = "NCOVR.shp"

    Extract the HR (homicide rates) data in the 70's, 80's and 90's from the dataframe
    and make it the dependent variable for the regression. Note that the data can also
    be passed in the long format instead of wide format (i.e. a vector with n*t rows
    and a single column for the dependent variable and a matrix of dimension n*txk
    for the independent variables).

    >>> y_name = ['HR70','HR80','HR90']
    >>> y = db[y_name]

    Extract RD and PS in the same time periods from the dataframe to be used as
    independent variables in the regression. 
    IMPORTANT: The data must be organized in a way that all time periods of a 
    given variable are side-by-side and in the correct time order.
    That is: x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    By default a vector of ones will be added to the independent variables passed in.

    >>> x_names = ['RD70','RD80','RD90','PS70','PS80','PS90']
    >>> x = db[x_names]

    Since we are running a spatial model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations. To do that, 
    we can open an already existing gal file or create a new one. In this case, 
    we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = 'r'
    >>> name_w = "NCOVR.gal"

    We are all set with the preliminaries, we are good to run the model.

    >>> reg = ML_ErrorPooled(y, x, w)

    Once we have run the model, we can explore a little bit the output with 
    the command print(reg.summary).    
    """

    def __init__(
        self,
        y,
        x,
        w,
        method="LU", 
        epsilon=0.0000001,
        slx_lags=0,
        slx_vars="all",
        nonspat_diag=True,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
    ):
        self.title = "ML POOLED SPATIAL ERROR MODEL (SEM)"
        bigy, bigx, self.name_y, self.name_x, w, warn, self.slx_lags, self.title, kx, kwx, self.t = prepare_panel(
            y, x, w, name_y, name_x, slx_lags, slx_vars, self.title, add_constant=True)
        any(set_warn(self, i) for i in warn)

        self.k = bigx.shape[1]
        
        BaseML_ErrorPooled.__init__(self, bigy, bigx, w, method=method, epsilon=epsilon)
        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.slx_lags = slx_lags
        self.name_w = USER.set_name_w(name_w, w)
        
        self.name_x.append("lambda")

        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['o'] + ['x'] * (kx-1) + ['wx'] * kwx + ['lambda']
        self.output['regime'], self.output['equation'] = (0, 0)

        self.aic = DIAG.akaike(reg=self)
        self.schwarz = DIAG.schwarz(reg=self)
        self.other_top = _nonspat_top(self, ml=True)     

        if nonspat_diag:
            self.bsk = BSK_tests(self, w, which=['LMC_RE'])
            other_end = "\nDIAGNOSTIC TESTS\n"
            other_end += f"{"-" * 84 }\n"
            other_end += "TEST                             DF        VALUE           PROB\n"
            for i in range(len(self.bsk)):
                other_end += f"{self.bsk['Test'][i]:30s} {self.bsk['df'][i]:3d}   {self.bsk['Statistic'][i]:12.4f}       {self.bsk['p-value'][i]:8.5f}\n"  

        output(reg=self, vm=vm, robust=False, other_end=other_end, latex=latex)

class BaseGM_ErrorRE(RegressionPropsY):

    '''
    Base GMM method for a spatial random effects panel model based on
    Kapoor, Kelejian and Prucha (2007) :cite:`KKP2007`.

    Parameters
    ----------
    y          : array
                 n*tx1 array for dependent variable
    x          : array
                 Two dimensional array with n*t rows and one column for each
                 independent (exogenous) variable
                 (note: must already include constant term)
    w          : spatial weights object
                 Spatial weights matrix
    full_weights: boolean
                  Considers different weights for each of the 6 moment
                  conditions if True or only 2 sets of weights for the
                  first 3 and the last 3 monent conditions if False (default)

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
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    vm           : array
                   Variance covariance matrix (kxk)
    """
    '''

    def __init__(self, y, x, w, full_weights=False):

        # 1a. OLS --> \tilde{\delta}
        ols = BaseOLS(y=y, x=x)
        self.x, self.y, self.n, self.k, self.xtx = ols.x, ols.y, ols.n, ols.k, ols.xtx
        N = w.n
        T = y.shape[0] // N
        moments, trace_w2 = self._moments_kkp(w.sparse, ols.u, 0)
        lambda1, sig_v = optim_moments(moments, all_par=True)
        Tw = sps.kron(sps.identity(T), w.sparse)
        ub = Tw.dot(ols.u)
        ulu = ols.u - lambda1 * ub
        Q1 = sps.kron(np.ones((T, T)) / T, sps.identity(N))
        sig_1 = (np.dot(ulu.T, Q1.dot(ulu)) / N).item()

        Xi_a = sps.diags([(sig_v * sig_v) / (T - 1), sig_1 * sig_1])
        if full_weights:
            Tau = self._get_Tau(w.sparse, trace_w2)
        else:
            Tau = sps.identity(3)
        Xi = sps.kron(Xi_a, Tau)
        moments_b, _ = self._moments_kkp(w.sparse, ols.u, 1, trace_w2)
        G = np.vstack((np.hstack((moments[0], np.zeros((3, 1)))), moments_b[0]))
        moments6 = [G, np.vstack((moments[1], moments_b[1]))]
        lambda2, sig_vb, sig_1b = optim_moments(
            moments6, vcX=Xi.toarray(), all_par=True, start=[lambda1, sig_v, sig_1]
        )

        # 2a. reg -->\hat{betas}
        theta = 1 - np.sqrt(sig_vb) / np.sqrt(sig_1b)
        gls_w = sps.identity(N * T) - theta * Q1

        # With omega
        xs = gls_w.dot(get_spFilter(w, lambda2, x))
        ys = gls_w.dot(get_spFilter(w, lambda2, y))
        ols_s = BaseOLS(y=ys, x=xs)
        self.predy = spdot(self.x, ols_s.betas)
        self.u = self.y - self.predy
        self.vm = ols_s.vm  
        self.betas = np.vstack((ols_s.betas, lambda2, sig_vb, sig_1b))
        self.e_filtered = self.u - lambda2 * sps.kron(sps.identity(T), w.sparse).dot(
            self.u
        )
        self.t = T

    def _moments_kkp(self, ws, u, i, trace_w2=None):
        """
        Compute G and g matrices for the KKP model.
        ...

        Parameters
        ----------
        ws          : Sparse matrix
                    Spatial weights sparse matrix
        u           : array
                    Residuals. nx1 array assumed to be aligned with w

        i       : integer
                    0 if Q0, 1 if Q1
        trace_w2    : float
                    trace of WW. Computed in 1st step and saved for step 2.
        Returns
        -------
        moments     : list
                    List of two arrays corresponding to the matrices 'G' and
                    'g', respectively.
        trace_w2    : float
                    trace of WW. Computed in 1st step and saved for step 2.
        """
        N = ws.shape[0]
        T = u.shape[0] // N

        """
        # NT matrix approach
        if i == 0:
            Q = sps.kron(sps.identity(T) - np.ones((T, T)) / T, sps.identity(N))
        else:
            Q = sps.kron(np.ones((T, T)) / T, sps.identity(N))
        Tw = sps.kron(sps.identity(T), ws)
        ub = Tw.dot(u)
        ubb = Tw.dot(ub)
        Qu = Q.dot(u)
        Qub = Q.dot(ub)
        Qubb = Q.dot(ubb)
        G11 = (2 * np.dot(u.T, Qub)).item()
        G12 = -(np.dot(ub.T, Qub)).item()
        G21 = (2 * np.dot(ubb.T, Qub)).item()
        G22 = -(np.dot(ubb.T, Qubb)).item()
        G31 = (np.dot(u.T, Qubb) + np.dot(ub.T, Qub)).item()
        G32 = -(np.dot(ub.T, Qubb)).item()
        if trace_w2 == None:
            trace_w2 = (ws.power(2)).sum()
        G23 = trace_w2
        if i == 0:
            G = np.array(
                [[G11, G12, N * (T - 1)], [G21, G22, G23*(T - 1)], [G31, G32, 0]]
            ) / (N * (T - 1))
        else:
            G = np.array(
                [
                    [G11, G12, 0, N],
                    [G21, G22, 0, G23],
                    [G31, G32, 0, 0],
                ]
            ) / N
        g1 = np.dot(u.T, Qu).item()
        g2 = np.dot(ub.T, Qub).item()
        g3 = np.dot(u.T, Qub).item()
        g = np.array([[g1, g2, g3]]).T / (N * (T - 1) ** (1 - i))
        """
        # TxN matrix approach
        u_mat = u.reshape((T, N))
        ub_mat = (ws @ u_mat.T).T
        ubb_mat = (ws @ ub_mat.T).T 
        
        u_means = u_mat.mean(axis=0)     
        ub_means = ub_mat.mean(axis=0) 
        ubb_means = ubb_mat.mean(axis=0)

        if i == 0:
            Qu = u_mat - u_means
            Qub = ub_mat - ub_means
            Qubb = ubb_mat - ubb_means
        else:
            Qu = np.tile(u_means, (T, 1))
            Qub = np.tile(ub_means, (T, 1))
            Qubb = np.tile(ubb_means, (T, 1))
        
        G11 = 2 * np.sum(u_mat * Qub)
        G12 = -np.sum(ub_mat * Qub)
        G21 = 2 * np.sum(ubb_mat * Qub)
        G22 = -np.sum(ubb_mat * Qubb)
        G31 = np.sum(u_mat * Qubb) + np.sum(ub_mat * Qub)
        G32 = -np.sum(ub_mat * Qubb)
        
        if trace_w2 is None:
            trace_w2 = (ws.power(2)).sum()
        G23 = trace_w2

        if i == 0:
            G = np.array([
                [G11, G12, N * (T - 1)], 
                [G21, G22, G23 * (T - 1)], 
                [G31, G32, 0]
            ]) / (N * (T - 1))
        else:
            G = np.array([
                [G11, G12, 0, N],
                [G21, G22, 0, G23],
                [G31, G32, 0, 0]
            ]) / N

        g1 = np.sum(u_mat * Qu)
        g2 = np.sum(ub_mat * Qub)
        g3 = np.sum(u_mat * Qub)
        g = np.array([[g1, g2, g3]]).T / (N * (T - 1) ** (1 - i))

        return [G, g], trace_w2

    def _get_Tau(self, ws, trace_w2):
        """
        Computes Tau as in :cite:`KKP2007`.
        ...

        Parameters
        ----------
        ws          : Sparse matrix
                    Spatial weights sparse matrix
        trace_w2    : float
                    trace of WW. Computed in 1st step of _moments_kkp
        """
        N = ws.shape[0]
        T12 = 2 * trace_w2 / N
        wtw = ws.T.dot(ws)
        T22 = wtw.power(2).sum()
        wtpw = ws.T + ws
        T23 = wtw.multiply(wtpw).sum()
        d_wwpwtw = ws.multiply(ws.T).sum(0) + wtw.diagonal()
        T33 = d_wwpwtw.sum()
        Tau = np.array([[2 * N, T12, 0], [T12, T22, T23], [0, T23, T33]]) / N
        return Tau
    
class GM_ErrorRE(BaseGM_ErrorRE, REGI.Regimes_Frame):

    '''
    GMM method for a spatial random effects panel model based on
    Kapoor, Kelejian and Prucha (2007) :cite:`KKP2007`.

    Parameters
    ----------
    y          : array or pandas DataFrame
                 n*tx1 or nxt array for dependent variable
    x          : array or pandas DataFrame
                 Two dimensional array or DF with n*t rows and k columns for
                 independent (exogenous) variable or n rows and k*t columns
                 (note, must not include a constant term)
    w          : spatial weights object
                 Spatial weights matrix, nxn
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged                 
    full_weights: boolean
                  Considers different weights for each of the 6 moment
                  conditions if True or only 2 sets of weights for the
                  first 3 and the last 3 moment conditions if False (default)
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string or list of strings
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
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)    
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    vm           : array
                   Variance covariance matrix (kxk)
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output

    Examples
    --------

    >>> import libpysal
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> from spreg import GM_ErrorRE
    
    In this example, we will use data on NCOVR US County Homicides (3085 areas) included in libpysal.

    >>> libpysal.examples.load_example('NCOVR')
    >>> db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    >>> name_ds = "NCOVR.shp"

    Extract the HR (homicide rates) data in the 70's, 80's and 90's from the dataframe
    and make it the dependent variable for the regression. Note that the data can also
    be passed in the long format instead of wide format (i.e. a vector with n*t rows
    and a single column for the dependent variable and a matrix of dimension n*txk
    for the independent variables).

    >>> y_name = ['HR70','HR80','HR90']
    >>> y = db[y_name]

    Extract RD and PS in the same time periods from the dataframe to be used as
    independent variables in the regression. Data must be
    organized in a way that all time periods of a given variable are side-by-side
    and in the correct time order.
    By default a vector of ones will be added to the independent variables passed in.

    >>> x_names = ['RD70','RD80','RD90','PS70','PS80','PS90']
    >>> x = db[x_names]

    Since we want to run a spatial error panel model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``NAT.shp``.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = 'r'
    >>> name_w = "NCOVR.gal"

    We are all set with the preliminaries, we are good to run the model.

    >>> reg = GM_ErrorRE(y, x, w)

    Once we have run the model, we can explore a little bit the output with 
    the command print(reg.summary).
 
    '''
    def __init__(
        self,
        y,
        x,
        w,
        slx_lags=0,
        slx_vars="all",
        full_weights=False,
        regimes=None,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        name_regimes=None,
        latex=False,
    ):
        self.title = "GMM SPATIAL ERROR PANEL MODEL - RANDOM EFFECTS (KKP)"
        bigy, bigx, self.name_y, self.name_x, w, warn, self.slx_lags, self.title, kx, kwx, self.t = prepare_panel(
            y, x, w, name_y, name_x, slx_lags, slx_vars, self.title, add_constant=True)
        any(set_warn(self, i) for i in warn)
 
        if regimes is not None:
            raise NotImplementedError("Regimes are not currently implemented for GM_ErrorRE.")
            """
            self.regimes = regimes
            self.name_regimes = USER.set_name_ds(name_regimes)
            regimes_l = self._set_regimes(w, bigy.shape[0])
            self.name_x_r = self.name_x
            x_constant, self.name_x, xtype = REGI.Regimes_Frame.__init__(
                self,
                x_constant,
                regimes_l,
                constant_regi=False,
                cols2regi="all",
                names=self.name_x,
            )
            """
        BaseGM_ErrorRE.__init__(self, bigy, bigx, w, full_weights=full_weights)
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_x.extend(["lambda", " sigma2_v", "sigma2_1"])
        self.name_w = USER.set_name_w(name_w, w)

        """ Yet to be implemented within new output frame
        if regimes is not None:
            self.kf += 3
            self.chow = REGI.Chow(self)
            self.title += " WITH REGIMES"
            regimes = True
        """
        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['o'] + ['x'] * (kx-1) + ['wx'] * kwx + ["lambda", " sigma2_v", "sigma2_1"]
        self.output['regime'], self.output['equation'] = (0, 0)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)

    def _set_regimes(self, w, n_rows):  # Place holder only from old code. Must add case for regime_err_sep = True
        self.constant_regi = "many"
        self.cols2regi = "all"
        self.regime_err_sep = False
        self.regimes_set = REGI._get_regimes_set(self.regimes)
        if len(self.regimes) == w.n:
            regimes_l = self.regimes * (n_rows // w.n)
        elif len(self.regimes) == n_rows:
            regimes_l = self.regimes
        else:
            raise Exception("The lenght of 'regimes' must be either equal to n or n*t.")
        return regimes_l

class BaseML_ErrorFE(RegressionPropsY, RegressionPropsVM):

    """
    Base ML method for a fixed effects spatial error model (note no consistency
    checks, diagnostics or constants added) :cite:`Elhorst2003`.

    Parameters
    ----------
    y         : array
                (n*t)x1 array for dependent variable
    x         : array
                Two dimensional array with n*t rows and one column for each
                independent (exogenous) variable
                (note: must already include constant term)
    w         : pysal W object
                Spatial weights matrix
    epsilon   : float
                tolerance criterion in mimimize_scalar function and
                inverse_product

    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    lam          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (no constant, excluding the lambda)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable, no constant
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
    """

    def __init__(self, y, x, w, epsilon=0.0000001):
        # set up main regression variables and spatial filters
        N = w.n
        NT = y.shape[0]
        self.t = NT // N
        self.k = x.shape[1]
        self.epsilon = epsilon
        # Demeaned variables
        self.y = demean_panel(y, N, self.t)
        self.x = demean_panel(x, N, self.t)
        # Big W matrix
        W = w.full()[0]
        Wsp = w.sparse
        Wsp_nt = sps.kron(sps.identity(self.t), Wsp, format="csr")
        # lag dependent variable
        ylag = spdot(Wsp_nt, self.y)
        xlag = spdot(Wsp_nt, self.x)

        # concentrated Log Likelihood
        I = sps.identity(N)
        res = minimize_scalar(
            self.err_c_loglik_sp,
            0.0,
            bounds=(-1.0, 1.0),
            args=(N, self.t, self.y, ylag, self.x, xlag, I, Wsp),
            method="bounded",
            options={"xatol": epsilon},
        )
        self.lam = res.x

        # compute full log-likelihood
        ln2pi = np.log(2.0 * np.pi)
        self.logll = (-res.fun - NT / 2.0 * ln2pi - NT / 2.0)
        # adjusting sigma2 by NT:
        self.logll += (NT/2.0)*np.log(NT)
        
        # b, residuals and predicted values
        ys = self.y - self.lam * ylag
        xs = self.x - self.lam * xlag
        xsxs = spdot(xs.T, xs)
        xsxsi = la.inv(xsxs)
        xsys = spdot(xs.T, ys)
        b = spdot(xsxsi, xsys)

        self.betas = np.vstack((b, self.lam))

        self.u = self.y - spdot(self.x, b)
        self.predy = self.y - self.u

        # residual variance
        self.e_filtered = self.u - self.lam * spdot(Wsp_nt, self.u)
        self.sig2 = spdot(self.e_filtered.T, self.e_filtered) / NT

        # variance-covariance matrix betas
        varb = self.sig2 * xsxsi

        # variance-covariance matrix lambda, sigma
        a = -self.lam * W
        spfill_diagonal(a, 1.0)
        ai = spinv(a)
        wai = spdot(Wsp, ai)
        tr1 = wai.diagonal().sum()

        wai2 = spdot(wai, wai)
        tr2 = wai2.diagonal().sum()

        waiTwai = spdot(wai.T, wai)
        tr3 = waiTwai.diagonal().sum()

        v1 = np.vstack((self.t * (tr2 + tr3), self.t * tr1 / self.sig2))
        v2 = np.vstack(
            (self.t * tr1 / self.sig2, NT / (2.0 * self.sig2 ** 2))
        )

        v = np.hstack((v1, v2))

        self.vm1 = la.inv(v)

        # create variance matrix for beta, lambda
        vv = np.hstack((varb, np.zeros((self.k, 1))))
        vv1 = np.hstack((np.zeros((1, self.k)), self.vm1[0, 0] * np.ones((1, 1))))

        self.vm = np.vstack((vv, vv1))
        self.varb = varb
        self.n = NT

    def err_c_loglik_sp(self, lam, n, t, y, ylag, x, xlag, I, Wsp):
        # concentrated log-lik for error model, no constants, LU
        if isinstance(lam, np.ndarray):
            if lam.shape == (1, 1):
                lam = lam[0][0]
        ys = y - lam * ylag
        xs = x - lam * xlag
        ysys = np.dot(ys.T, ys)
        xsxs = np.dot(xs.T, xs)
        xsxsi = la.inv(xsxs)
        xsys = np.dot(xs.T, ys)
        x1 = np.dot(xsxsi, xsys)
        x2 = np.dot(xsys.T, x1)
        ee = ysys - x2
        sig2 = ee[0][0]
        nlsig2 = (n * t / 2.0) * np.log(sig2)
        a = I - lam * Wsp
        LU = sps.linalg.splu(a.tocsc())
        jacob = t * np.sum(np.log(np.abs(LU.U.diagonal())))
        # this is the negative of the concentrated log lik for minimization
        clik = nlsig2 - jacob
        return clik

class ML_ErrorFE(BaseML_ErrorFE):

    """
    ML estimation of the fixed effects spatial error model with all results and
    diagnostics :cite:`Elhorst2003`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas object
                   nxt or (nxt)x1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, no constant
    w            : pysal W object
                   Spatial weights object
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged                   
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and
                   inverse_product
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
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)    
    betas        : array
                   k+1x1 array of estimated coefficients
    lam          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   (nxt)x1 array of residuals
    e_filtered   : array
                   (nxt)x1 array of spatially filtered residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (no constant, excluding the lambda)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable, including the constant
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
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    std_err      : array
                   1x(k+1) array of standard errors of the betas
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
    mean_mu_i    : float
                   Mean of the fixed effects (mu_i) across all observations
    mu_i         : array
                   nx1 array of the fixed effects (mu_i) for each observation

    Examples
    --------

    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spreg import ML_ErrorFE
    
    In this example, we will use data on NCOVR US County Homicides (3085 areas) included in libpysal.

    >>> libpysal.examples.load_example('NCOVR')
    >>> db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    >>> name_ds = "NCOVR.shp"

    Extract the HR (homicide rates) data in the 70's, 80's and 90's from the dataframe
    and make it the dependent variable for the regression. Note that the data can also
    be passed in the long format instead of wide format (i.e. a vector with n*t rows
    and a single column for the dependent variable and a matrix of dimension n*txk
    for the independent variables).

    >>> y_name = ['HR70','HR80','HR90']
    >>> y = db[y_name]

    Extract RD and PS in the same time periods from the dataframe to be used as
    independent variables in the regression. 
    IMPORTANT: The data must be organized in a way that all time periods of a 
    given variable are side-by-side and in the correct time order.
    That is: x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    By default a vector of ones will be added to the independent variables passed in.

    >>> x_names = ['RD70','RD80','RD90','PS70','PS80','PS90']
    >>> x = db[x_names]

    Since we are running a spatial model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations. To do that, 
    we can open an already existing gal file or create a new one. In this case, 
    we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = 'r'
    >>> name_w = "NCOVR.gal"

    We are all set with the preliminaries, we are good to run the model.

    >>> reg = ML_ErrorFE(y, x, w)

    Once we have run the model, we can explore a little bit the output with 
    the command print(reg.summary).    
    """

    def __init__(
        self,
        y,
        x,
        w,
        slx_lags=0,
        slx_vars="all",
        epsilon=0.0000001,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
    ):

        self.title = "ML SPATIAL ERROR PANEL MODEL - FIXED EFFECTS"
        bigy, bigx, self.name_y, self.name_x, w, warn, self.slx_lags, self.title, kx, kwx, self.t = prepare_panel(
            y, x, w, name_y, name_x, slx_lags, slx_vars, self.title, add_constant=False)
        any(set_warn(self, i) for i in warn)

        BaseML_ErrorFE.__init__(self, bigy, bigx, w, epsilon=epsilon)

        self.name_ds = USER.set_name_ds(name_ds)
        self.name_x.append("lambda")
        self.name_w = USER.set_name_w(name_w, w)

        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['o'] + ['x'] * (kx-1) + ['wx'] * kwx + ["lambda"]
        self.output['regime'], self.output['equation'] = (0, 0)

        self.aic = DIAG.akaike(reg=self)
        self.schwarz = DIAG.schwarz(reg=self)
        self.other_top = _nonspat_top(self, ml=True)      

        u_undemean = bigy - np.dot(bigx, self.betas[:-1])
        self.mu_i = u_undemean.reshape(self.t, w.n).mean(axis=0).reshape(-1, 1)        
        self.mean_mu_i = self.mu_i.mean()
        self.other_top += "%-20s:%12.4f\n" % (
            "Fixed-effects mean", self.mean_mu_i)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)

class BaseML_ErrorRE(RegressionPropsY, RegressionPropsVM):

    """
    Base ML method for a random effects spatial error model (note no
    consistency checks, diagnostics or constants added) :cite:`Elhorst2003`.

    Parameters
    ----------
    y         : array
                (n*t)x1 array for dependent variable
    x         : array
                Two dimensional array with n*t rows and one column for each
                independent (exogenous) variable
                (note: must already include constant term)
    w         : pysal W object
                Spatial weights matrix
    epsilon   : float
                tolerance criterion in mimimize_scalar function and
                inverse_product

    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    lam          : float
                   estimate of spatial autoregressive coefficient
    sig2_u       : float
                   Sigma squared for random effects
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (no constant, excluding the lambda)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable, no constant
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+2 x k+2)
    vm1          : array
                   Variance covariance matrix (k+3 x k+3) includes sigma2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    """

    def __init__(self, y, x, w, epsilon=0.0000001):
        # set up main regression variables and spatial filters
        N = w.n
        NT = y.shape[0]
        self.t = NT // N
        self.k = x.shape[1]
        self.epsilon = epsilon

        self.y = y
        self.x = x
        # Big W matrix
        W = w.full()[0]
        Wsp = w.sparse
        
        y_reshaped = self.y.reshape(self.t, N)
        ylag = (y_reshaped @ Wsp.T).reshape(-1, 1)
        
        x_reshaped = self.x.reshape(self.t, N, self.k)
        xlag = np.zeros_like(self.x)
        for i in range(self.k):
            xlag[:, i] = (x_reshaped[:, :, i] @ Wsp.T).reshape(-1)

        I = np.identity(N)
        evals, evecs = la.eig(W)
        
        y_mean_small = y_reshaped.mean(axis=0).reshape(N, 1)        
        x_mean_small = x_reshaped.mean(axis=0) # N x k

        res = minimize(
            self.err_c_loglik_ord,
            (0.0, 0.1),
            bounds=((-0.999, 0.999), (1e-8, 1e+09)),
            method="L-BFGS-B",
            args=(
                evals,
                evecs,
                N,
                self.t,
                self.y,
                self.x,
                ylag,
                xlag,
                y_mean_small, 
                x_mean_small,
                I,
                Wsp,
            ),
            options={'ftol': 1e-9}
        )
        self.lam, self.phi = res.x
        self.phi = self.phi**2

        ln2pi = np.log(2.0 * np.pi)
        self.logll = (
            -res.fun - NT / 2.0 * ln2pi - NT / 2.0
        )

        I = np.identity(N)
        B = I - self.lam * W

        BB = np.dot(B.T, B)
        BB_inv = la.inv(BB)

        Omega_between = (self.t * self.phi * I) + BB_inv

        eig_O, vec_O = la.eigh(Omega_between)
        P = np.dot(vec_O, np.dot(np.diag(eig_O**(-0.5)), vec_O.T))

        pr = P - B  
        
        pr_y_mean = np.dot(pr, y_mean_small)
        pr_x_mean = np.dot(pr, x_mean_small)

        term_y = np.tile(pr_y_mean.T, (self.t, 1)).reshape(-1, 1)
        term_x = np.tile(pr_x_mean, (self.t, 1)).reshape(-1, self.k)

        yrand = self.y + term_y
        xrand = self.x + term_x

        ys = yrand - self.lam * ylag
        xs = xrand - self.lam * xlag

        # Final OLS
        xsxs = spdot(xs.T, xs)
        xsxsi = la.inv(xsxs)
        xsys = spdot(xs.T, ys)
        b = spdot(xsxsi, xsys)

        self.u = self.y - spdot(self.x, b)
        self.predy = self.y - self.u

        # residual variance
        self.e_filtered = ys - spdot(xs, b)
        self.sig2 = spdot(self.e_filtered.T, self.e_filtered) / NT

        varb = self.sig2 * xsxsi
        self.sig2_u = self.phi * self.sig2

        self.betas = np.vstack((b, self.lam, self.sig2_u))

        # variance-covariance matrix lambda, sigma
        a = -self.lam * W
        spfill_diagonal(a, 1.0)
        aTai = la.inv(spdot(a.T, a))
        wa_aw = spdot(W.T, a) + spdot(a.T, W)
        gamma = spdot(wa_aw, aTai)
        vi = la.inv(self.t * np.sqrt(self.phi) * I + aTai)
        sigma = spdot(vi, aTai)

        tr1 = gamma.diagonal().sum()
        tr2 = vi.diagonal().sum()
        tr3 = sigma.diagonal().sum()

        sigma_gamma = spdot(sigma, gamma)
        tr4 = sigma_gamma.diagonal().sum()

        sigma_vi = spdot(sigma, vi)
        tr5 = sigma_vi.diagonal().sum()

        sigma_gamma_vi = spdot(sigma_gamma, vi)
        tr6 = sigma_gamma_vi.diagonal().sum()

        sigma_gamma_sigma = spdot(sigma_gamma, sigma)
        tr7 = sigma_gamma_sigma.diagonal().sum()

        v1 = np.vstack(
            (
                (self.t - 1) / 2 * tr1 ** 2 + 1 / 2 * tr4 ** 2,
                self.t / (2 * self.sig2) * tr6,
                (self.t - 1) / (2 * self.sig2) * tr1 + 1 / (2 * self.sig2) * tr7,
            )
        )
        v2 = np.vstack(
            (
                self.t / (2 * self.sig2) * tr6,
                self.t ** 2 / (2.0 * self.sig2 ** 2) * tr2 ** 2,
                self.t / (2.0 * self.sig2 ** 2) * tr5,
            )
        )
        v3 = np.vstack(
            (
                (self.t - 1) / (2 * self.sig2) * tr1 + 1 / (2 * self.sig2) * tr7,
                self.t / (2.0 * self.sig2 ** 2) * tr5,
                1 / (2.0 * self.sig2 ** 2) * ((self.t - 1) * N + tr3 ** 2),
            )
        )

        v = np.hstack((v1, v2, v3))

        vm1 = np.linalg.inv(v)

        # create variance matrix for beta, lambda
        vv = np.hstack((varb, np.zeros((self.k, 2))))
        vv1 = np.hstack((np.zeros((2, self.k)), vm1[:2, :2]))

        self.vm = np.vstack((vv, vv1))
        self.n = NT

    def err_c_loglik_ord(self, 
        lam_phi, evals, evecs, n, t, bigy, bigx, ylag, xlag, y_mean, x_mean, I, Wsp
    ):
        lam, phi = lam_phi
        cvals = t * phi ** 2 + 1 / (1 - lam * evals) ** 2
        P = np.dot(np.diag(cvals ** (-0.5)), evecs.T)
        
        pr = P - (I - lam * Wsp)
        
        pr_y = np.dot(pr, y_mean)
        pr_x = np.dot(pr, x_mean)
        
        term_y = np.tile(pr_y, (t, 1))
        term_x = np.tile(pr_x, (t, 1))

        yrand = bigy + term_y
        xrand = bigx + term_x
        
        ys = yrand - lam * ylag
        xs = xrand - lam * xlag
        
        ysys = np.dot(ys.T, ys)
        xsxs = np.dot(xs.T, xs)
        xsxsi = la.inv(xsxs)
        xsys = np.dot(xs.T, ys)
        x1 = np.dot(xsxsi, xsys)
        x2 = np.dot(xsys.T, x1)
        ee = ysys - x2
        
        sig2 = ee[0][0] / (n * t)
        nlsig2 = (n * t / 2.0) * np.log(sig2)

        revals = t * phi ** 2 * (1 - lam * evals) ** 2
        phi_jacob = 1 / 2 * np.log(1 + revals).sum()

        jacob = t * np.log(1 - lam * evals).sum()
        if isinstance(jacob, complex):
            jacob = jacob.real
            
        clik = nlsig2 + phi_jacob - jacob
        return clik
    
class ML_ErrorRE(BaseML_ErrorRE):

    """
    ML estimation of the random effects spatial error model with all results and
    diagnostics :cite:`Elhorst2003`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas object
                   nxt or (nxt)x1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, no constant
    w            : pysal W object
                   Spatial weights object
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged                   
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and
                   inverse_product
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
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)    
    betas        : array
                   kx1 array of estimated coefficients
    lam          : float
                   estimate of spatial autoregressive coefficient
    sig2_u       : float
                   Sigma squared for random effects
    u            : array
                   (nxt)x1 array of residuals
    e_filtered   : array
                   (nxt)x1 array of spatially filtered residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (no constant, excluding the lambda)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable, including the constant
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+2 x k+2), all coefficients
    vm1          : array
                   Variance covariance matrix (k+3 x k+3), includes sig2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    aic          : float
                   Akaike information criterion
    schwarz      : float
                   Schwarz criterion
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    utu          : float
                   Sum of squared residuals
    std_err      : array
                   1x(k+1) array of standard errors of the betas
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

    Examples
    --------

    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spreg import ML_ErrorRE
    
    In this example, we will use data on NCOVR US County Homicides (3085 areas) included in libpysal.

    >>> libpysal.examples.load_example('NCOVR')
    >>> db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    >>> name_ds = "NCOVR.shp"

    Extract the HR (homicide rates) data in the 70's, 80's and 90's from the dataframe
    and make it the dependent variable for the regression. Note that the data can also
    be passed in the long format instead of wide format (i.e. a vector with n*t rows
    and a single column for the dependent variable and a matrix of dimension n*txk
    for the independent variables).

    >>> y_name = ['HR70','HR80','HR90']
    >>> y = db[y_name]

    Extract RD and PS in the same time periods from the dataframe to be used as
    independent variables in the regression. 
    IMPORTANT: The data must be organized in a way that all time periods of a 
    given variable are side-by-side and in the correct time order.
    That is: x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    By default a vector of ones will be added to the independent variables passed in.

    >>> x_names = ['RD70','RD80','RD90','PS70','PS80','PS90']
    >>> x = db[x_names]

    Since we are running a spatial model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations. To do that, 
    we can open an already existing gal file or create a new one. In this case, 
    we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = 'r'
    >>> name_w = "NCOVR.gal"

    We are all set with the preliminaries, we are good to run the model.

    >>> reg = ML_ErrorRE(y, x, w)

    Once we have run the model, we can explore a little bit the output with 
    the command print(reg.summary).    
    """

    def __init__(
        self,
        y,
        x,
        w,
        slx_lags=0,
        slx_vars="all",
        epsilon=0.0000001,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
    ):
        self.title = "ML SPATIAL ERROR PANEL MODEL (SEM) - RANDOM EFFECTS"
        bigy, bigx, self.name_y, self.name_x, w, warn, self.slx_lags, self.title, kx, kwx, self.t = prepare_panel(
            y, x, w, name_y, name_x, slx_lags, slx_vars, self.title, add_constant=True)
        any(set_warn(self, i) for i in warn)

        BaseML_ErrorRE.__init__(self, bigy, bigx, w, epsilon=epsilon)
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_x.extend(["lambda", "sigma2_u"])
        self.name_w = USER.set_name_w(name_w, w)
        self.aic = DIAG.akaike(reg=self)
        self.schwarz = DIAG.schwarz(reg=self)

        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['o'] + ['x'] * (kx-1) + ['wx'] * kwx + ["lambda", "sigma2_u"]
        self.output['regime'], self.output['equation'] = (0, 0)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)

class BaseML_LagFE(RegressionPropsY, RegressionPropsVM):

    """
    Base ML method for a fixed effects spatial lag model (note no consistency
    checks, diagnostics or constants added) :cite:`Elhorst2003`.

    Parameters
    ----------
    y         : array
                (n*t)x1 array for dependent variable
    x         : array
                Two dimensional array with n*t rows and one column for each
                independent (exogenous) variable
                (note: must already include constant term)
    w         : pysal W object
                Spatial weights matrix
    epsilon   : float
                tolerance criterion in mimimize_scalar function and
                inverse_product

    Attributes
    ----------
    betas        : array
                   (k+1)x1 array of estimated coefficients (rho last)
    rho          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (no constant, excluding the rho)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable, no constant
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
    """

    def __init__(self, y, x, w, epsilon=0.0000001):
        # set up main regression variables and spatial filters
        N = w.n
        NT = y.shape[0]
        self.t = NT // N
        self.k = x.shape[1]
        self.epsilon = epsilon
        # Demeaned variables
        self.y = demean_panel(y, N, self.t)
        self.x = demean_panel(x, N, self.t)
        # Big W matrix
        W = w.full()[0]
        Wsp = w.sparse
        Wsp_nt = sps.kron(sps.identity(self.t), Wsp, format="csr")
        # lag dependent variable
        ylag = spdot(Wsp_nt, self.y)
        # b0, b1, e0 and e1
        xtx = spdot(self.x.T, self.x)
        xtxi = la.inv(xtx)
        xty = spdot(self.x.T, self.y)
        xtyl = spdot(self.x.T, ylag)
        b0 = spdot(xtxi, xty)
        b1 = spdot(xtxi, xtyl)
        e0 = self.y - spdot(self.x, b0)
        e1 = ylag - spdot(self.x, b1)

        # concentrated Log Likelihood
        I = sps.identity(N)
        res = minimize_scalar(
            self.lag_c_loglik_sp,
            0.0,
            bounds=(-0.99, 0.99),
            args=(N, self.t, e0, e1, I, Wsp),
            method="bounded",
            options={"xatol": epsilon},
        )
        self.rho = res.x[0][0]

        # compute full log-likelihood, including constants
        ln2pi = np.log(2.0 * np.pi)
        llik = -res.fun - NT / 2.0 * ln2pi - NT / 2.0
        self.logll = llik[0][0]

        # b, residuals and predicted values
        b = b0 - self.rho * b1
        self.betas = np.vstack((b, self.rho))  # rho added as last coefficient
        self.u = e0 - self.rho * e1
        self.predy = self.y - self.u

        xb = spdot(self.x, b)

        self.predy_e = inverse_prod(
            Wsp_nt, xb, self.rho, inv_method="power_exp", threshold=epsilon
        )
        self.e_pred = self.y - self.predy_e

        # residual variance
        self._cache = {}
        self.sig2 = spdot(self.u.T, self.u) / NT

        # information matrix
        a = -self.rho * W
        spfill_diagonal(a, 1.0)
        ai = spinv(a)
        wai = spdot(Wsp, ai)
        tr1 = wai.diagonal().sum()  # same for sparse and dense

        wai2 = spdot(wai, wai)
        tr2 = wai2.diagonal().sum()

        waiTwai = spdot(wai.T, wai)
        tr3 = waiTwai.diagonal().sum()

        wai_nt = sps.kron(sps.identity(self.t), wai, format="csr")
        wpredy = spdot(wai_nt, xb)
        xTwpy = spdot(x.T, wpredy)

        waiTwai_nt = sps.kron(sps.identity(self.t), waiTwai, format="csr")
        wTwpredy = spdot(waiTwai_nt, xb)
        wpyTwpy = spdot(xb.T, wTwpredy)

        # order of variables is beta, rho, sigma2
        v1 = np.vstack((xtx / self.sig2, xTwpy.T / self.sig2, np.zeros((1, self.k))))
        v2 = np.vstack(
            (
                xTwpy / self.sig2,
                self.t * (tr2 + tr3) + wpyTwpy / self.sig2,
                self.t * tr1 / self.sig2,
            )
        )
        v3 = np.vstack(
            (
                np.zeros((self.k, 1)),
                self.t * tr1 / self.sig2,
                NT / (2.0 * self.sig2 ** 2),
            )
        )

        v = np.hstack((v1, v2, v3))

        self.vm1 = la.inv(v)  # vm1 includes variance for sigma2
        self.vm = self.vm1[:-1, :-1]  # vm is for coefficients only
        self.varb = la.inv(np.hstack((v1[:-1], v2[:-1])))
        self.k += 1  # add one to k to account for rho
        self.n = NT

    def lag_c_loglik_sp(self, rho, n, t, e0, e1, I, Wsp):
        # concentrated log-lik for lag model, sparse algebra
        if isinstance(rho, np.ndarray):
            if rho.shape == (1, 1):
                rho = rho[0][0]
        er = e0 - rho * e1
        sig2 = spdot(er.T, er) / (n * t)
        nlsig2 = (n * t / 2.0) * np.log(sig2)
        a = I - rho * Wsp
        LU = sps.linalg.splu(a.tocsc())
        jacob = t * np.sum(np.log(np.abs(LU.U.diagonal())))
        clike = nlsig2 - jacob
        return clike

class ML_LagFE(BaseML_LagFE):

    """
    ML estimation of the fixed effects spatial lag model with all results and
    diagnostics :cite:`Elhorst2003`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas object
                   nxt or (nxt)x1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, no constant
    w            : pysal W object
                   Spatial weights object
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged                   
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and
                   inverse_product
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
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)    
    betas        : array
                   (k+1)x1 array of estimated coefficients (rho last)
    rho          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (no constant, excluding the rho)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable, no constant
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
    predy_e      : array
                   predicted values from reduced form
    e_pred       : array
                   prediction errors using reduced form predicted values
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
    std_err      : array
                   1x(k+1) array of standard errors of the betas
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
    mean_mu_i    : float
                   Mean of the fixed effects (mu_i) across observations
    mu_i         : array
                   nx1 array of the fixed effects (mu_i) for each observation
                   
    Examples
    --------

    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spreg import ML_LagFE
    
    In this example, we will use data on NCOVR US County Homicides (3085 areas) included in libpysal.

    >>> libpysal.examples.load_example('NCOVR')
    >>> db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    >>> name_ds = "NCOVR.shp"

    Extract the HR (homicide rates) data in the 70's, 80's and 90's from the dataframe
    and make it the dependent variable for the regression. Note that the data can also
    be passed in the long format instead of wide format (i.e. a vector with n*t rows
    and a single column for the dependent variable and a matrix of dimension n*txk
    for the independent variables).

    >>> y_name = ['HR70','HR80','HR90']
    >>> y = db[y_name]

    Extract RD and PS in the same time periods from the dataframe to be used as
    independent variables in the regression. 
    IMPORTANT: The data must be organized in a way that all time periods of a 
    given variable are side-by-side and in the correct time order.
    That is: x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    By default a vector of ones will be added to the independent variables passed in.

    >>> x_names = ['RD70','RD80','RD90','PS70','PS80','PS90']
    >>> x = db[x_names]

    Since we are running a spatial model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations. To do that, 
    we can open an already existing gal file or create a new one. In this case, 
    we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = 'r'
    >>> name_w = "NCOVR.gal"

    We are all set with the preliminaries, we are good to run the model.

    >>> reg = ML_LagFE(y, x, w)

    Once we have run the model, we can explore a little bit the output with 
    the command print(reg.summary).    
    """

    def __init__(
        self,
        y,
        x,
        w,
        slx_lags=0,
        slx_vars="all",
        epsilon=0.0000001,
        spat_impacts="simple",        
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
    ):
        self.title = "ML SPATIAL LAG PANEL - FIXED EFFECTS"

        bigy, bigx, self.name_y, self.name_x, w, warn, self.slx_lags, self.title, kx, kwx, self.t = prepare_panel(
            y, x, w, name_y, name_x, slx_lags, slx_vars, self.title, add_constant=False)
        any(set_warn(self, i) for i in warn)

        BaseML_LagFE.__init__(self, bigy, bigx, w, epsilon=epsilon)

        self.name_ds = USER.set_name_ds(name_ds)
        name_ylag = USER.set_name_yend_sp(self.name_y)
        self.name_x.append(name_ylag)
        self.name_w = USER.set_name_w(name_w, w)
        self.aic = DIAG.akaike(reg=self)
        self.schwarz = DIAG.schwarz(reg=self)
        
        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['x'] * kx + ['wx'] * kwx + ['rho']
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top, self.other_mid, other_end = ("", "", "")
        self.other_top += _nonspat_top(self, ml=True)

        u_undemean = bigy - np.dot(bigx, self.betas[:-1])
        self.mu_i = u_undemean.reshape(self.t, w.n).mean(axis=0).reshape(-1, 1)
        self.mean_mu_i = self.mu_i.mean()
        self.other_top += "%-20s:%12.4f\n" % (
            "Fixed-effects mean", self.mean_mu_i)        
        if spat_impacts:
            self.sp_multipliers, impacts_str = _summary_impacts(self, w, spat_impacts, slx_lags,slx_vars)
            other_end += impacts_str
        output(reg=self, vm=vm, other_end=other_end, latex=latex)

class BaseML_LagRE(RegressionPropsY, RegressionPropsVM):

    """
    Base ML method for a random effects spatial lag model (note no consistency
    checks, diagnostics or constants added) :cite:`Elhorst2003`.

    Parameters
    ----------
    y         : array
                (n*t)x1 array for dependent variable
    x         : array
                Two dimensional array with n*t rows and one column for each
                independent (exogenous) variable
                (note: must already include constant term)
    w         : pysal W object
                Spatial weights matrix
    epsilon   : float
                tolerance criterion in mimimize_scalar function and
                inverse_product

    Attributes
    ----------
    betas        : array
                   (k+2)x1 array of estimated coefficients (rho and phi last)
    rho          : float
                   estimate of spatial autoregressive coefficient
    phi          : float
                   estimate of weight attached to the cross-sectional component
                   phi^2 = sig2 / (t*sig2_u + sig2)
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (no constant, excluding the rho and phi)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable, no constant
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+2 x k+2)
    vm1          : array
                   Variance covariance matrix (k+3 x k+3) includes sigma2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    predy_e      : array
                   predicted values from reduced form
    e_pred       : array
                   prediction errors using reduced form predicted values
    """

    def __init__(self, bigy, bigx, w, epsilon=0.0000001):
        # set up main regression variables and spatial filters
        N = w.n
        NT = bigy.shape[0]
        self.t = NT // N
        self.k = bigx.shape[1]
        self.epsilon = epsilon
        # Big W matrix
        W = w.full()[0]
        Wsp = w.sparse
        Wsp_nt = sps.kron(sps.identity(self.t), Wsp, format="csr")
        # Set up parameters
        converge = 1
        criteria = 0.0000001
        i = 0
        itermax = 100
        self.rho = 0.1
        self.phi = 0.1 #Baltagi's transformation factor
        I = sps.identity(N)
        xtx = spdot(bigx.T, bigx)
        xtxi = la.inv(xtx)
        xty = spdot(bigx.T, bigy)
        b = spdot(xtxi, xty)

        # Iterative procedure
        while converge > criteria and i < itermax:
            phiold = self.phi
            res_phi = minimize_scalar(
                self.phi_c_loglik,
                0.1,
                bounds=(1e-8, 0.99999999),
                args=(self.rho, b, bigy, bigx, N, self.t, Wsp_nt),
                method="bounded",
                options={"xatol": epsilon},
            )
            self.phi = res_phi.x[0][0]
            # Demeaned variables
            self.y = demean_panel(bigy, N, self.t, phi=self.phi)
            self.x = demean_panel(bigx, N, self.t, phi=self.phi)
            # lag dependent variable
            ylag = spdot(Wsp_nt, self.y)
            # b0, b1, e0 and e1
            xtx = spdot(self.x.T, self.x)
            xtxi = la.inv(xtx)
            xty = spdot(self.x.T, self.y)
            xtyl = spdot(self.x.T, ylag)
            b0 = spdot(xtxi, xty)
            b1 = spdot(xtxi, xtyl)
            e0 = self.y - spdot(self.x, b0)
            e1 = ylag - spdot(self.x, b1)
            res_rho = minimize_scalar(
                self.lag_c_loglik_sp,
                0.0,
                bounds=(-0.99, 0.99),
                args=(N, self.t, e0, e1, I, Wsp),
                method="bounded",
                options={"xatol": epsilon},
            )
            self.rho = res_rho.x[0][0]
            b = b0 - self.rho * b1
            i += 1
            converge = np.abs(phiold - self.phi)

        # compute full log-likelihood, including constants
        ln2pi = np.log(2.0 * np.pi) 
        phi_jacob = (N / 2.0) * np.log(self.phi ** 2)
        llik = (-res_rho.fun + phi_jacob 
            - NT / 2.0 * ln2pi 
            - NT / 2.0)
        self.logll = llik[0][0] 

        # b, residuals and predicted values
        self.betas = np.vstack((b, self.rho, self.phi))
        self.u = e0 - self.rho * e1
        self.predy = self.y - self.u
        xb = spdot(self.x, b)

        self.predy_e = inverse_prod(
            Wsp_nt, xb, self.rho, inv_method="power_exp", threshold=epsilon
        )
        self.e_pred = self.y - self.predy_e

        # residual variance
        self._cache = {}
        self.sig2 = spdot(self.u.T, self.u) / NT

        # information matrix
        a = -self.rho * W
        spfill_diagonal(a, 1.0)
        ai = spinv(a)
        wai = spdot(W, ai)
        tr1 = wai.diagonal().sum() 

        wai2 = spdot(wai, wai)
        tr2 = wai2.diagonal().sum()

        waiTwai = spdot(wai.T, wai)
        tr3 = waiTwai.diagonal().sum()

        wai_nt = sps.kron(sps.identity(self.t), wai, format="csr")
        wpredy = spdot(wai_nt, xb)
        xTwpy = spdot(self.x.T, wpredy)

        waiTwai_nt = sps.kron(sps.identity(self.t), waiTwai, format="csr")
        wTwpredy = spdot(waiTwai_nt, xb)
        wpyTwpy = spdot(xb.T, wTwpredy)

        # order of variables is beta, rho, sigma2
        v1 = np.vstack((xtx / self.sig2, xTwpy.T / self.sig2, np.zeros((2, self.k))))
        v2 = np.vstack(
            (
                xTwpy / self.sig2,
                self.t * (tr2 + tr3) + wpyTwpy / self.sig2,
                -tr1 / self.sig2,
                self.t * tr1 / self.sig2,
            )
        )
        v3 = np.vstack(
            (
                np.zeros((self.k, 1)),
                -tr1 / self.sig2,
                N * (1 + 1 / self.phi ** 2),
                -N / self.sig2,
            )
        )
        v4 = np.vstack(
            (
                np.zeros((self.k, 1)),
                self.t * tr1 / self.sig2,
                -N / self.sig2 ** 2,
                NT / (2.0 * self.sig2 ** 2),
            )
        )

        v = np.hstack((v1, v2, v3, v4))

        self.vm1 = la.inv(v)  # vm1 includes variance for sigma2
        self.vm = self.vm1[:-1, :-1]  # vm is for coefficients and phi
        self.varb = la.inv(np.hstack((v1[:-2], v2[:-2])))
        self.k += 1 # add one to k to account for rho
        self.n = NT


class ML_LagRE(BaseML_LagRE):

    """
    ML estimation of the random effects spatial lag model with all results and
    diagnostics :cite:`Elhorst2003`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas object
                   nxt or (nxt)x1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, excluding the constant
    w            : pysal W object
                   Spatial weights object
    spat_impacts : string or list
                   Include average direct impact (ADI), average indirect impact (AII),
                    and average total impact (ATI) in summary results.
                    Options are 'simple', 'full', 'power', 'all' or None.
                    See sputils.spmultiplier for more information.
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and
                   inverse_product
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

    Attributes
    ----------
    betas        : array
                   (k+2)x1 array of estimated coefficients (rho and phi last)
    rho          : float
                   estimate of spatial autoregressive coefficient
    phi          : float
                   estimate of weight attached to the cross-sectional component
                   phi^2 = sig2 / (t*sig2_u + sig2)
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (excluding the phi)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+2 x k+2)
    vm1          : array
                   Variance covariance matrix (k+3 x k+3) includes sigma2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
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
                   1x(k+1) array of standard errors of the betas
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

    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spreg import ML_LagRE
    
    In this example, we will use data on NCOVR US County Homicides (3085 areas) included in libpysal.

    >>> libpysal.examples.load_example('NCOVR')
    >>> db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    >>> name_ds = "NCOVR.shp"

    Extract the HR (homicide rates) data in the 70's, 80's and 90's from the dataframe
    and make it the dependent variable for the regression. Note that the data can also
    be passed in the long format instead of wide format (i.e. a vector with n*t rows
    and a single column for the dependent variable and a matrix of dimension n*txk
    for the independent variables).

    >>> y_name = ['HR70','HR80','HR90']
    >>> y = db[y_name]

    Extract RD and PS in the same time periods from the dataframe to be used as
    independent variables in the regression. 
    IMPORTANT: The data must be organized in a way that all time periods of a 
    given variable are side-by-side and in the correct time order.
    That is: x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    By default a vector of ones will be added to the independent variables passed in.

    >>> x_names = ['RD70','RD80','RD90','PS70','PS80','PS90']
    >>> x = db[x_names]

    Since we are running a spatial model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations. To do that, 
    we can open an already existing gal file or create a new one. In this case, 
    we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = 'r'
    >>> name_w = "NCOVR.gal"

    We are all set with the preliminaries, we are good to run the model.

    >>> reg = ML_LagRE(y, x, w)

    Once we have run the model, we can explore a little bit the output with 
    the command print(reg.summary).    
    """

    def __init__(
        self,
        y,
        x,
        w,
        slx_lags=0,
        slx_vars="all",
        epsilon=0.0000001,
        spat_impacts="simple",        
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
    ):

        self.title = "ML SPATIAL LAG PANEL - RANDOM EFFECTS"

        bigy, bigx, self.name_y, self.name_x, w, warn, self.slx_lags, self.title, kx, kwx, self.t = prepare_panel(
            y, x, w, name_y, name_x, slx_lags, slx_vars, self.title, add_constant=True)
        any(set_warn(self, i) for i in warn)

        BaseML_LagRE.__init__(self, bigy, bigx, w, epsilon=epsilon)

        self.name_ds = USER.set_name_ds(name_ds)
        name_ylag = USER.set_name_yend_sp(self.name_y)
        self.name_x.append(name_ylag)  # rho changed to last position
        self.name_x.append("phi")  # error variance parameter
        self.name_w = USER.set_name_w(name_w, w)
        self.aic = DIAG.akaike(reg=self)
        self.schwarz = DIAG.schwarz(reg=self)
        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['o'] + ['x'] * (kx-1) + ['wx'] * kwx + ['rho', 'phi']
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top, self.other_mid, other_end = ("", "", "")
        self.other_top += _nonspat_top(self, ml=True)
        if spat_impacts:
            self.sp_multipliers, impacts_str = _summary_impacts(self, w, spat_impacts, slx_lags,slx_vars)
            other_end += impacts_str
        output(reg=self, vm=vm, other_end=other_end, latex=latex)


    def lag_c_loglik_sp(self, rho, n, t, e0, e1, I, Wsp):
        # concentrated log-lik for lag model, sparse algebra
        if isinstance(rho, np.ndarray):
            if rho.shape == (1, 1):
                rho = rho[0][0]
        er = e0 - rho * e1
        sig2 = spdot(er.T, er) / (n * t)
        nlsig2 = (n * t / 2.0) * np.log(sig2)
        a = I - rho * Wsp
        LU = sps.linalg.splu(a.tocsc())
        jacob = t * np.sum(np.log(np.abs(LU.U.diagonal())))
        clike = nlsig2 - jacob
        return clike


    def phi_c_loglik(self, phi, rho, beta, bigy, bigx, n, t, W_nt):
        # Demeaned variables
        y = demean_panel(bigy, n, t, phi=phi)
        x = demean_panel(bigx, n, t, phi=phi)
        # Lag dependent variable
        ylag = spdot(W_nt, y)
        er = y - rho * ylag - spdot(x, beta)
        sig2 = spdot(er.T, er)
        nlsig2 = (n * t / 2.0) * np.log(sig2)
        nphi2 = (n / 2.0) * np.log(phi ** 2)
        clike = nlsig2 - nphi2
        return clike


