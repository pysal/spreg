"""
ML Estimation of Spatial Error Model
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu"

import libpysal
import numpy as np
import multiprocessing as mp
from . import regimes as REGI
from . import user_output as USER
from . import diagnostics as DIAG
from .utils import set_warn, get_lags
from .sputils import sphstack
from .ml_error import BaseML_Error
from platform import system
import pandas as pd
from .output import output, _nonspat_top

__all__ = ["ML_Error_Regimes"]


class ML_Error_Regimes(BaseML_Error, REGI.Regimes_Frame):

    """
    ML estimation of the spatial error model with regimes (note no consistency 
    checks, diagnostics or constants added); :cite:`Anselin1988`

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
    w            : Sparse matrix
                   Spatial weights sparse matrix 
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX-Error type.                          
    method       : string
                   if 'full', brute force calculation (full matrix expressions)
                   if 'ord', Ord eigenvalue computation
                   if 'LU', LU sparse matrix decomposition
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and inverse_product
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    regime_lag_sep: boolean
                    Always False, kept for consistency in function call, ignored.
    cores        : boolean
                   Specifies if multiprocessing is to be used
                   Default: no multiprocessing, cores = False
                   Note: Multiprocessing may not work on all platforms.
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
    name_regimes : string
                   Name of regimes variable for use in output
    latex        : boolean
                   Specifies if the table with the coefficients' results and their inference is to be printed in LaTeX format

    Attributes
    ----------
    output       : dataframe
                   regression results pandas dataframe
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   (k+1)x1 array of estimated coefficients (lambda last)
    lam          : float
                   estimate of spatial autoregressive coefficient
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
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
                   (including the constant, excluding the rho)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    method       : string
                   log Jacobian method.
                   if 'full': brute force (full matrix computations)
                   if 'ord', Ord eigenvalue computation
                   if 'LU', LU sparse matrix decomposition
    epsilon      : float
                   tolerance criterion used in minimize_scalar function and inverse_product
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+1 x k+1), all coefficients
    vm1          : array
                   variance covariance matrix for lambda, sigma (2 x 2)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2         : float
                   Sigma squared used in computations
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    logll        : float
                   maximized log-likelihood (including constant terms)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
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
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regimes variable for use in output
    title        : string
                   Name of the regression method used
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

                   * 'many': a vector of ones is appended to x and considered different per regime (default).
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

    Open data baltim.dbf using pysal and create the variables matrices and weights matrix.

    >>> import numpy as np
    >>> import libpysal
    >>> from libpysal.examples import load_example
    >>> from libpysal.weights import Queen
    >>> from spreg import ML_Error_Regimes
    >>> import geopandas as gpd
    >>> np.set_printoptions(suppress=True) #prevent scientific format
    >>> baltimore = load_example('Baltimore')
    >>> db = libpysal.io.open(baltimore.get_path("baltim.dbf"),'r')
    >>> df = gpd.read_file(baltimore.get_path("baltim.shp"))
    >>> ds_name = "baltim.dbf"
    >>> y_name = "PRICE"
    >>> y = np.array(db.by_col(y_name)).T
    >>> y.shape = (len(y),1)
    >>> x_names = ["NROOM","AGE","SQFT"]
    >>> x = np.array([db.by_col(var) for var in x_names]).T
    >>> w = Queen.from_dataframe(df)
    >>> w_name = "baltim_q.gal"
    >>> w.transform = 'r'    

    Since in this example we are interested in checking whether the results vary
    by regimes, we use CITCOU to define whether the location is in the city or 
    outside the city (in the county):

    >>> regimes = db.by_col("CITCOU")

    Now we can run the regression with all parameters:

    >>> mlerr = ML_Error_Regimes(y,x,regimes,w=w,name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name,name_regimes="CITCOU")
    >>> np.around(mlerr.betas, decimals=4)
    array([[-2.076 ],
           [ 4.8615],
           [-0.0295],
           [ 0.3355],
           [32.3457],
           [ 2.8708],
           [-0.2401],
           [ 0.799 ],
           [ 0.6   ]])
    >>> "{0:.6f}".format(mlerr.lam)
    '0.599951'
    >>> "{0:.6f}".format(mlerr.mean_y)
    '44.307180'
    >>> "{0:.6f}".format(mlerr.std_y)
    '23.606077'
    >>> np.around(mlerr.vm1, decimals=4)
    array([[  0.0053,  -0.3643],
           [ -0.3643, 465.3559]])
    >>> np.around(np.diag(mlerr.vm), decimals=4)
    array([58.7121,  2.5036,  0.0074,  0.0659, 81.9796,  3.2676,  0.0124,
            0.0514,  0.0053])
    >>> np.around(mlerr.sig2, decimals=4)
    array([[215.554]])
    >>> "{0:.6f}".format(mlerr.logll)
    '-872.860883'
    >>> "{0:.6f}".format(mlerr.aic)
    '1761.721765'
    >>> "{0:.6f}".format(mlerr.schwarz)
    '1788.536630'
    >>> mlerr.title
    'MAXIMUM LIKELIHOOD SPATIAL ERROR - REGIMES (METHOD = full)'
    """

    def __init__(
        self,
        y,
        x,
        regimes,
        w=None,
        slx_lags=0,
        constant_regi="many",
        cols2regi="all",
        method="full",
        epsilon=0.0000001,
        regime_err_sep=False,
        regime_lag_sep=False,
        cores=False,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        name_regimes=None,
        latex=False,
    ):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        self.constant_regi = constant_regi
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_w = USER.set_name_w(name_w, w)
        regimes, name_regimes = USER.check_reg_list(regimes, name_regimes, n)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.n = n
        self.y = y

        x_constant, name_x, warn = USER.check_constant(x, name_x, just_rem=True)
        set_warn(self, warn)
        name_x = USER.set_name_x(name_x, x_constant, constant=True)

        if slx_lags >0:
            lag_x = get_lags(w, x_constant, slx_lags)
            x_constant = np.hstack((x_constant, lag_x))
            name_x += USER.set_name_spatial_lags(name_x, slx_lags)

        self.name_x_r = USER.set_name_x(name_x, x_constant)

        cols2regi = REGI.check_cols2regi(constant_regi, cols2regi, x_constant)
        self.cols2regi = cols2regi        
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, self.n, x.shape[1])
        self.regime_err_sep = regime_err_sep

        if regime_err_sep == True:
            if set(cols2regi) == set([True]):
                self._error_regimes_multi(
                    y,
                    x_constant,
                    regimes,
                    w,
                    slx_lags,
                    cores,
                    method,
                    epsilon,
                    cols2regi,
                    vm,
                    name_x,
                    latex,
                )
            else:
                raise Exception(
                    "All coefficients must vary across regimes if regime_err_sep = True."
                )
        else:
            x_constant = sphstack(np.ones((x_constant.shape[0], 1)), x_constant)
            name_x = USER.set_name_x(name_x, x_constant)
            regimes_att = {}
            regimes_att["x"] = x_constant
            regimes_att["regimes"] = regimes
            regimes_att["cols2regi"] = cols2regi
            x, name_x, x_rlist = REGI.Regimes_Frame.__init__(
                self,
                x_constant,
                regimes,
                constant_regi=None,
                cols2regi=cols2regi,
                names=name_x,
                rlist=True
            )
            BaseML_Error.__init__(
                self,
                y=y,
                x=x,
                w=w,
                method=method,
                epsilon=epsilon,
                regimes_att=regimes_att,
            )

            self.title = "ML SPATIAL ERROR"
            if slx_lags >0:
                self.title += " WITH SLX (SLX-Error)"
            self.title += " - REGIMES (METHOD = " + method + ")"

            self.name_x = USER.set_name_x(name_x, x, constant=True)
            self.name_x.append("lambda")
            self.kf += 1  # Adding a fixed k to account for lambda.
            self.chow = REGI.Chow(self)
            self.aic = DIAG.akaike(reg=self)
            self.schwarz = DIAG.schwarz(reg=self)
            self.output = pd.DataFrame(self.name_x, columns=['var_names'])
            self.output['var_type'] = ['x'] * (len(self.name_x) - 1) + ['lambda']
            self.output['regime'] = x_rlist + ['_Global']
            self.output['equation'] = 0
            self.other_top = _nonspat_top(self, ml=True)
            output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)

    def _error_regimes_multi(
        self, y, x, regimes, w, slx_lags, cores, method, epsilon, cols2regi, vm, name_x, latex
    ):

        regi_ids = dict(
            (r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set
        )
        results_p = {}
        """
        for r in self.regimes_set:
            if system() == 'Windows':
                is_win = True
                results_p[r] = _work_error(*(y,x,regi_ids,r,w,method,epsilon,self.name_ds,self.name_y,name_x+['lambda'],self.name_w,self.name_regimes))
            else:
                pool = mp.Pool(cores)
                results_p[r] = pool.apply_async(_work_error,args=(y,x,regi_ids,r,w,method,epsilon,self.name_ds,self.name_y,name_x+['lambda'],self.name_w,self.name_regimes, ))
                is_win = False
        """
        x_constant, name_x = REGI.check_const_regi(self, x, name_x, regi_ids)
        self.name_x_r = name_x
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(
                    _work_error,
                    args=(
                        y,
                        x_constant,
                        regi_ids,
                        r,
                        w,
                        slx_lags,
                        method,
                        epsilon,
                        self.name_ds,
                        self.name_y,
                        name_x + ["lambda"],
                        self.name_w,
                        self.name_regimes,
                    ),
                )
            else:
                results_p[r] = _work_error(
                    *(
                        y,
                        x_constant,
                        regi_ids,
                        r,
                        w,
                        slx_lags,
                        method,
                        epsilon,
                        self.name_ds,
                        self.name_y,
                        name_x + ["lambda"],
                        self.name_w,
                        self.name_regimes,
                    )
                )

        self.kryd = 0
        self.kr = len(cols2regi) + 1
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * self.kr, 1), float)
        self.u = np.zeros((self.n, 1), float)
        self.predy = np.zeros((self.n, 1), float)
        self.e_filtered = np.zeros((self.n, 1), float)
        self.name_y, self.name_x = [], []
        """
        if not is_win:
            pool.close()
            pool.join()
        """
        if cores:
            pool.close()
            pool.join()

        results = {}
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
            self.e_filtered[
                regi_ids[r],
            ] = results[r].e_filtered
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            results[r].other_top = _nonspat_top(results[r], ml=True)
            self.output = pd.concat([self.output, pd.DataFrame({'var_names': results[r].name_x,
                                                                'var_type': ['x'] * (len(results[r].name_x) - 1) + ['lambda'],
                                                                'regime': r, 'equation': r})], ignore_index=True)
            counter += 1
        self.chow = REGI.Chow(self)
        self.multi = results
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


def _work_error(
    y, x, regi_ids, r, w, slx_lags, method, epsilon, name_ds, name_y, name_x, name_w, name_regimes
):
    w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    model = BaseML_Error(y=y_r, x=x_r, w=w_r, method=method, epsilon=epsilon)
    set_warn(model, warn)
    model.w = w_r
    model.title = "ML SPATIAL ERROR"
    if slx_lags >0:
        model.title += " WITH SLX (SLX-Error)"
    model.title += " - REGIME " + str(r) + " (METHOD = " + method + ")"
    model.name_ds = name_ds
    model.name_y = "%s_%s" % (str(r), name_y)
    model.name_x = ["%s_%s" % (str(r), i) for i in name_x]
    model.name_w = name_w
    model.name_regimes = name_regimes
    model.aic = DIAG.akaike(reg=model)
    model.schwarz = DIAG.schwarz(reg=model)
    return model


def _test():
    import doctest

    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == "__main__":
    _test()
    import numpy as np
    import libpysal as ps

    db = ps.io.open(ps.examples.get_path("baltim.dbf"), "r")
    ds_name = "baltim.dbf"
    y_name = "PRICE"
    y = np.array(db.by_col(y_name)).T
    y.shape = (len(y), 1)
    x_names = ["NROOM", "NBATH", "PATIO", "FIREPL", "AC", "GAR", "AGE", "LOTSZ", "SQFT"]
    x = np.array([db.by_col(var) for var in x_names]).T
    ww = ps.io.open(ps.examples.get_path("baltim_q.gal"))
    w = ww.read()
    ww.close()
    w_name = "baltim_q.gal"
    w.transform = "r"
    regimes = db.by_col("CITCOU")

    model = ML_Error_Regimes(
        y,
        x,
        regimes,
        w=w,
        method="full",
        name_y=y_name,
        name_x=x_names,
        name_w=w_name,
        name_ds=ds_name,
        regime_err_sep=True,
        constant_regi="many",
        name_regimes="CITCOU",
    )
    print(model.output)
    print(model.summary)
