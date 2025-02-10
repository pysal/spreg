"""Probit regression class and diagnostics."""

__author__ = "Luc Anselin lanselin@gmail.com, Pedro V. Amaral pedrovma@gmail.com"

import numpy as np
import numpy.linalg as la
import scipy.optimize as op
import pandas as pd
from scipy.stats import norm, chi2
from libpysal import weights
chisqprob = chi2.sf
import scipy.sparse as SP
from . import user_output as USER
from . import diagnostics as DIAGNOSTICS  
from . import diagnostics_probit as PROBDIAG   
from .output import output, _probit_out
from .utils import spdot, spbroadcast, set_warn

__all__ = ["Probit"]

class BaseProbit():
    """
    Probit class to do the core estimation

    Parameters
    ----------

    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent binary variable
    optim       : string
                  Optimization method.
                  Default: 'newton' (Newton-Raphson).
                  Alternatives: 'ncg' (Newton-CG), 'bfgs' (BFGS algorithm)
    bstart      : list with starting values for betas
    maxiter     : int
                  Maximum number of iterations until optimizer stops

    Attributes
    ----------
    x           : array
                  Two dimensional array with n rows and one column for each
                  independent (exogenous) variable, including the constant
    xmean       : array
                  kx1 vector with means of explanatory variables
                  (for use in slopes)
    y           : array
                  nx1 array of dependent variable
    optim       : string
                  optimization method
    maxiter     : int
                  maximum number of iterations
    q           : array
                  nx1 array of transformed dependent variable 2*y - 1
    betas       : array
                  kx1 array with estimated coefficients
    bstart      : list with starting values
    predy       : array
                  nx1 array of predicted y values (probabilities)
    n           : int
                  Number of observations
    k           : int
                  Number of variables
    vm          : array
                  Variance-covariance matrix (kxk)
    logl        : float
                  Log-Likelihhod of the estimation
    xb          : predicted value of linear index
                  nx1 array
    predybin    : predicted value as binary
                  =1 for predy > 0.5
    phiy        : normal density at xb (phi function)
                  nx1 array
    u_naive     : naive residuals y - predy
                  nx1 array
    u_gen       : generalized residuals
                  nx1 array
    warning     : boolean
                  if True Maximum number of iterations exceeded or gradient
                  and/or function calls not changing.

    Examples - needs updating
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> import spreg
    >>> np.set_printoptions(suppress=True) #prevent scientific format
    >>> dbf = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array([dbf.by_col('CRIME')]).T
    >>> x = np.array([dbf.by_col('INC'), dbf.by_col('HOVAL')]).T
    >>> x = np.hstack((np.ones(y.shape),x))
    >>> model = spreg.probit.BaseProbit((y>40).astype(float), x)
    >>> print(np.around(model.betas, decimals=6))
    [[ 3.353811]
     [-0.199653]
     [-0.029514]]

    >>> print(np.around(model.vm, decimals=6))
    [[ 0.852814 -0.043627 -0.008052]
     [-0.043627  0.004114 -0.000193]
     [-0.008052 -0.000193  0.00031 ]]

    """

    def __init__(self, y, x, optim="newton",bstart=False,maxiter=100):
        self.y = y
        self.q = 2* self.y - 1
        self.x = x
        self.xmean =  np.mean(self.x,axis=0).reshape(-1,1)
        self.n, self.k = x.shape
        self.optim = optim
        self.bstart = bstart
        self.maxiter = maxiter

        par_est, self.warning = self.par_est()
        self.betas = par_est[0].reshape(-1,1)

        self.logl = -float(par_est[1])
        H = self.hessian(self.betas)
        self.vm = - la.inv(H)

        # predicted values
        self.xb = self.x @ self.betas
        self.predy = norm.cdf(self.xb)
        self.predybin = (self.predy > 0.5) * 1
        self.phiy = norm.pdf(self.xb)
        
        # residuals
        self.u_naive = self.y - self.predy
        Phi_prod = self.predy * (1.0 - self.predy)
        self.u_gen = self.phiy * (self.u_naive / Phi_prod)


    def par_est(self):

        if self.bstart:
            if len(self.bstart) != self.k:
                raise Exception("Incompatible number of parameters in starting values")
            else:
                start = np.array(self.bstart).reshape(-1,1)
        else:
            xtx = self.x.T @ self.x
            xty = self.x.T @ self.y
            start = la.solve(xtx,xty)

        flogl = lambda par: -self.ll(par)

        if self.optim == "newton":
            fgrad = lambda par: self.gradient(par)
            fhess = lambda par: self.hessian(par)
            par_hat = newton(flogl, start, fgrad, fhess, self.maxiter)
            warn = par_hat[2]
        else:
            fgrad = lambda par: -self.gradient(par)
            if self.optim == "bfgs":
                par_hat = op.fmin_bfgs(flogl, start, fgrad, full_output=1, disp=0)
                warn = par_hat[6]
            if self.optim == "ncg":
                fhess = lambda par: -self.hessian(par)
                par_hat = op.fmin_ncg(
                    flogl, start, fgrad, fhess=fhess, full_output=1, disp=0
                )
                warn = par_hat[5]
        if warn > 0:
            warn = True
        else:
            warn = False
        return par_hat, warn

    def ll(self, par):
        beta = np.reshape(np.array(par), (self.k, 1))
        q = 2 * self.y - 1
        qxb = q * spdot(self.x, beta)
        ll = sum(np.log(norm.cdf(qxb)))
        return ll

    def gradient(self, par):
        beta = np.reshape(np.array(par), (self.k, 1))
        q = 2 * self.y - 1
        qxb = q * spdot(self.x, beta)
        lamb = q * norm.pdf(qxb) / norm.cdf(qxb)
        gradient = spdot(lamb.T, self.x)[0]
        return gradient

    def hessian(self, par):
        beta = np.reshape(np.array(par), (self.k, 1))
        q = 2 * self.y - 1
        xb = spdot(self.x, beta)
        qxb = q * xb
        lamb = q * norm.pdf(qxb) / norm.cdf(qxb)
        hessian = spdot(self.x.T, spbroadcast(self.x, -lamb * (lamb + xb)))
        return hessian


class Probit(BaseProbit):

    """
    Classic non-spatial Probit and spatial diagnostics. The class includes a
    printout that formats all the results and tests in a nice format.

    The diagnostics for spatial dependence currently implemented are:

    * Pinkse Error :cite:`Pinkse2004`

    * Kelejian and Prucha Moran's I :cite:`Kelejian2001`

    * Pinkse & Slade Error :cite:`Pinkse1998`

    Parameters
    ----------

    x           : numpy.ndarray or pandas object
                  nxk array of independent variables (assumed to be aligned with y)
    y           : numpy.ndarray or pandas.Series
                  nx1 array of dependent binary variable
    w           : W
                  PySAL weights instance aligned with y
    slx_lags    : integer
                  Number of spatial lags of X to include in the model specification.
                  If slx_lags>0, the specification becomes of the SLX type.
    slx_vars    : variables to be lagged when slx_lags > 0
                  default = "All", otherwise a list with Booleans indicating which
                  variables must be lagged (True) or not (False)
    optim       : string
                  Optimization method.
                  Default: 'newton' (Newton-Raphson).
                  Alternatives: 'ncg' (Newton-CG), 'bfgs' (BFGS algorithm)
    bstart      : list
                  list with starting values for betas, default = False
    scalem      : string
                  Method to calculate the scale of the marginal effects.
                  Default: 'phimean' (Mean of individual marginal effects)
                  Alternative: 'xmean' (Marginal effects at variables mean)
    predflag    : flag to print prediction table
    maxiter     : int
                  Maximum number of iterations until optimizer stops
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

    x           : array
                  Two dimensional array with n rows and one column for each
                  independent (exogenous) variable, including the constant
    xmean       : array
                  kx1 vector with means of explanatory variables
                  (for use in slopes)
    y           : array
                  nx1 array of dependent variable
    w           : spatial weights object
    optim       : string
                  optimization method
    predflag    : flag to print prediction table (predtable)
    maxiter     : int
                  maximum number of iterations
    q           : array
                  nx1 array of transformed dependent variable 2*y - 1
    betas       : array
                  kx1 array with estimated coefficients
    bstart      : list with starting values for betas or False
    predy       : array
                  nx1 array of predicted y values (probabilities)
    n           : int
                  Number of observations
    k           : int
                  Number of variables
    vm          : array
                  Variance-covariance matrix (kxk)
    logl        : float
                  Log-Likelihhod of the estimation
    xb          : predicted value of linear index
                  nx1 array
    predybin    : predicted value as binary
                  =1 for predy > 0.5
    phiy        : normal density at xb (phi function)
                  nx1 array
    u_naive     : naive residuals y - predy
                  nx1 array
    u_gen       : generalized residuals
                  nx1 array
    warning     : boolean
                  if True Maximum number of iterations exceeded or gradient
                  and/or function calls not changing.
    std_err     : standard errors of estimates
    z_stat      : list of tuples
                  z statistic; each tuple contains the pair (statistic,
                  p-value), where each is a float
    predtable   : dictionary
                  includes margins and cells of actual and predicted
                  values for discrete choice model
    fit         : a dictionary containing various measures of fit
                  TPR    : true positive rate (sensitivity, recall, hit rate)
                  TNR    : true negative rate (specificity, selectivity)
                  PREDPC : accuracy, percent correctly predicted
                  BA     : balanced accuracy
    predpc      : float
                  Percent of y correctly predicted (legacy)
    LRtest      : dictionary
                   contains the statistic for the null model (L0), the LR test(likr), 
                   the degrees of freedom (df) and the p-value (pvalue)
    L0          : likelihood of null model
    LR          : tuple (legacy)
                  Likelihood Ratio test of all coefficients = 0
                  (test statistics, p-value)
    mcfadrho    : McFadden rho statistics of fit
    scale       : float
                  Scale of the marginal effects.
    slopes      : array
                  Marginal effects of the independent variables (k-1x1)
    slopes_vm   : array
                  Variance-covariance matrix of the slopes (k-1xk-1)
    slopes_std_err : estimates of standard errors of marginal effects
    slopes_z_stat  : tuple with z-statistics and p-values for marginal effects
    Pinkse_error: array with statistic and p-value
                  Lagrange Multiplier test against spatial error correlation.
                  Implemented as presented in :cite:`Pinkse2004`
    KP_error    : array with statistic and p-value
                  Moran's I type test against spatial error correlation.
                  Implemented as presented in :cite:`Kelejian2001`
    PS_error    : array with statistic and p-value
                  Lagrange Multiplier test against spatial error correlation.
                  Implemented as presented in :cite:`Pinkse1998`
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
    data we read into arrays that ``spreg`` understands and ``libpysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import libpysal
    >>> np.set_printoptions(suppress=True) #prevent scientific format

    Open data on Columbus neighborhood crime (49 areas) using libpysal.io.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> dbf = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')

    Extract the CRIME column (crime) from the DBF file and make it the
    dependent variable for the regression. Note that libpysal requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept. Since we want to run a probit model and for this
    example we use the Columbus data, we also need to transform the continuous
    CRIME variable into a binary variable. As in :cite:`McMillen1992`, we define
    y = 1 if CRIME > 40.

    >>> y = np.array([dbf.by_col('CRIME')]).T
    >>> y = (y>40).astype(float)

    Extract HOVAL (home values) and INC (income) vectors from the DBF to be used as
    independent variables in the regression.  Note that libpysal requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
    independent variables passed in.

    >>> names_to_extract = ['INC', 'HOVAL']
    >>> x = np.array([dbf.by_col(name) for name in names_to_extract]).T

    Since we want to the test the probit model for spatial dependence, we need to
    specify the spatial weights matrix that includes the spatial configuration of
    the observations into the error component of the model. To do that, we can open
    an already existing gal file or create a new one. In this case, we will use
    ``columbus.gal``, which contains contiguity relationships between the
    observations in the Columbus dataset we are using throughout this example.
    Note that, in order to read the file, not only to open it, we need to
    append '.read()' at the end of the command.

    >>> w = libpysal.io.open(libpysal.examples.get_path("columbus.gal"), 'r').read()

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. In libpysal, this
    can be easily performed in the following way:

    >>> w.transform='r'

    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> from spreg import Probit
    >>> model = Probit(y, x, w=w, name_y='crime', name_x=['income','home value'], name_ds='columbus', name_w='columbus.gal')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them.

    >>> np.around(model.betas, decimals=6)
    array([[ 3.353811],
           [-0.199653],
           [-0.029514]])

    >>> np.around(model.vm, decimals=6)
    array([[ 0.852814, -0.043627, -0.008052],
           [-0.043627,  0.004114, -0.000193],
           [-0.008052, -0.000193,  0.00031 ]])

    Since we have provided a spatial weigths matrix, the diagnostics for
    spatial dependence have also been computed. We can access them and their
    p-values individually:

    >>> tests = np.array([['Pinkse_error','KP_error','PS_error']])
    >>> stats = np.array([[model.Pinkse_error[0],model.KP_error[0],model.PS_error[0]]])
    >>> pvalue = np.array([[model.Pinkse_error[1],model.KP_error[1],model.PS_error[1]]])
    >>> print(np.hstack((tests.T,np.around(np.hstack((stats.T,pvalue.T)),6))))
    [['Pinkse_error' '3.131719' '0.076783']
     ['KP_error' '1.721312' '0.085194']
     ['PS_error' '2.558166' '0.109726']]

    Or we can easily obtain a full summary of all the results nicely formatted and
    ready to be printed simply by typing 'print model.summary'

    """

    def __init__(
        self,
        y,
        x,
        w=None,
        slx_lags=0,
        slx_vars = "All",
        optim="newton",
        bstart=False,
        scalem="phimean",
        predflag = False,
        maxiter=100,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        spat_diag=False,
        latex=False,
    ):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        x_constant, name_x, warn = USER.check_constant(x, name_x)
        set_warn(self, warn)
        self.name_x = USER.set_name_x(name_x, x_constant)

        w_required=False
        if spat_diag:
            w_required=True
        w = USER.check_weights(w, y, w_required=w_required, slx_lags=slx_lags)

        if slx_lags > 0:
            x_constant,self.name_x = USER.flex_wx(w,x=x_constant,name_x=self.name_x,constant=True,
                                            slx_lags=slx_lags,slx_vars=slx_vars)            

        BaseProbit.__init__(
            self, y=y, x=x_constant, optim=optim, bstart=bstart, maxiter=maxiter
        )
        self.title = "CLASSIC PROBIT ESTIMATOR"
        if slx_lags > 0:
           self.title += " WITH SPATIALLY LAGGED X (SLX)"        
        self.slx_lags = slx_lags
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_w = USER.set_name_w(name_w, w)

        self.scalem = scalem
        self.predflag = predflag  # flag to print prediction table
        self.w = w   # note, not sparse

        # standard errors and z, p-values
        self.std_err = DIAGNOSTICS.se_betas(self)
        self.z_stat = DIAGNOSTICS.t_stat(self,z_stat=True)

        # truepos, trueneg, predpc - measures of fit
        self.predtable = PROBDIAG.pred_table(self)
        self.fit = PROBDIAG.probit_fit(self)
        self.predpc = self.fit["PREDPC"]
        self.LRtest = PROBDIAG.probit_lrtest(self)
        self.L0 = self.LRtest["L0"]
        self.LR = (self.LRtest["likr"],self.LRtest["p-value"])
        self.mcfadrho = PROBDIAG.mcfad_rho(self)

        # impact measures
        scale,slopes,slopes_vm,slopes_std_err,slopes_z_stat = PROBDIAG.probit_ape(self)
        self.scale = scale
        self.slopes = slopes
        self.slopes_vm = slopes_vm
        self.slopes_std_err = slopes_std_err
        self.slopes_z_stat = slopes_z_stat

        # tests for spatial autocorrelation
        if spat_diag:
            self.Pinkse_error,self.KP_error,self.PS_error = PROBDIAG.sp_tests(regprob=self)

        #output
        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['o'] + ['x'] * (len(self.name_x)-1)
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top, self.other_mid, other_end = _probit_out(reg=self, spat_diag=spat_diag)
        output(reg=self, vm=vm, robust=None, other_end=other_end, latex=latex)


def newton(flogl, start, fgrad, fhess, maxiter):

    """
    Calculates the Newton-Raphson method

    Parameters
    ----------

    flogl       : lambda
                  Function to calculate the log-likelihood
    start       : array
                  kx1 array of starting values
    fgrad       : lambda
                  Function to calculate the gradient
    fhess       : lambda
                  Function to calculate the hessian
    maxiter     : int
                  Maximum number of iterations until optimizer stops
    """
    warn = 0
    iteration = 0
    par_hat0 = start
    m = 1
    while iteration < maxiter and m >= 1e-04:
        H = -la.inv(fhess(par_hat0))
        g = fgrad(par_hat0).reshape(start.shape)
        Hg = np.dot(H, g)
        par_hat0 = par_hat0 + Hg
        iteration += 1
        m = np.dot(g.T, Hg)
    if iteration == maxiter:
        warn = 1
    logl = flogl(par_hat0)
    return (par_hat0, logl, warn)


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

    dbf = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    y = np.array([dbf.by_col("CRIME")]).T
    var_x = ["INC", "HOVAL"]
    x = np.array([dbf.by_col(name) for name in var_x]).T
    w = libpysal.io.open(libpysal.examples.get_path("columbus.gal"), "r").read()
    w.transform = "r"
    probit1 = Probit(
        (y > 40).astype(float),
        x,
        w=w,
        name_x=var_x,
        name_y="CRIME",
        name_ds="Columbus",
        name_w="columbus.dbf",
    )
    print(probit1.summary)

