"""
Estimation of Nonlinear SLX Model
"""

__author__ = "Luc Anselin lanselin@gmail.com, \
              Pedro V. Amaral pedrovma@gmail.com"

import numpy as np
import pandas as pd
from scipy.sparse import coo_array,csr_array
from scipy.optimize import minimize
from libpysal.cg import KDTree
from . import user_output as USER
import numpy.linalg as la
from .utils import make_wnslx, RegressionPropsY, set_warn
from .output import output, _nslx_out
from .diagnostics import log_likelihood,akaike,schwarz
from itertools import compress

__all__ = ["NSLX"]


class BaseNSLX(RegressionPropsY):
    '''
    Estimation of the nonlinear SLX model (note no consistency
    checks, diagnostics or constants added) - inverse distance
    power function and negative exponential distance function supported

    Parameters
    ----------
    y         : n by 1 numpy array with dependent variable
    x         : n by k array with explanatory variables, includes constant
    xw        : n by h array with selected columns of X that will have the
                W(alpha) transformation applied to them
    w         : list of sparse CSR arrays to use in lag transformation, if
                list has a single element, same weights applied for all
    transform : tuple of transformations, either "exponential" or "power"
                when same transformation applies to all, tuple is a single
                element tuple (default is "power")
    var_flag  : flag for analytical computation of variance-covariance matrix
                passed from NSLX
    verbose   : option for nonlinear optimization, either False (default) or
                True
    options   : options specific to scipy minimize, such as {"disp":True}
                (see scipy minimize docs)

    Attributes
    ----------
    y         : n by 1 numpy array with dependent variable
    x         : n by k array with explanatory variables, includes constant
    xw        : n by h array with selected columns of X that will have the
                W(alpha) transformation applied to them
    w         : list of sparse CSR arrays to use in lag transformation, if
                list has a single element, same weights applied for all
    n         : number of observations
    k         : number of explanatory variables in X (includes constant)
    transform : tuple of transformations, either "power" or "exponential"
                when same transformation applies to all, tuple is a single
                element tuple (default is "power")
    verbose   : option for nonlinear optimization, either False (default) or
                True
    options   : options specific to scipy minimize, such as {"disp":True}
                (see scipy minimize docs)
    betas     : numpy array with parameter estimates
    utu       : sum of squared residuals
    ihess     : inverse of Hessian matrix
    sign      : estimate of residual variance (divided by n)
    sig2      : same as sign
    vm      : coefficient variance-covariance matrix (sign x ihess)
    predy     : vector of predicted values
    u         : vector of residuals

    '''

    def __init__(self,y,x,xw,w,transform,var_flag,verbose,options):
        self.y = y
        self.x = x
        self.xw = xw
        self.n = self.x.shape[0]
        h = self.xw.shape[1]
        kk = self.x.shape[1]
        self.k = kk + h

        self.w = w
        self.transform = transform
        self.verbose = verbose
        self.options = options
              
        h = self.xw.shape[1]

        xty = self.x.T @ self.y
        xtx = self.x.T @ self.x
        b0 = la.solve(xtx,xty)

        alpha0 = np.ones((h,1))   # initial value
        g0 = np.vstack((b0,alpha0)) 
        gamma0 = g0.flatten()
        gradflag=0

        ssmin = minimize(nslxobj,gamma0,
                         args=(self.y,self.x,self.xw,self.w,self.transform,self.verbose),
                         options=self.options)   

        self.betas = ssmin.x
        self.utu = ssmin.fun
        self.sign = self.utu / self.n
        self.sig2 = self.sign

        # convergence criteria
        self.success = ssmin.success
        self.status = ssmin.status
        self.message = ssmin.message
        self.nit = ssmin.nit

        # compute predy
        b= self.betas[0:-h]
        alpha = self.betas[-h:]

        wx = nlmod(alpha,xw,w,transform,gradflag=0)
        wxs = np.sum(wx,axis=1).reshape(-1,1)
        xb = x @ b.reshape(-1,1)
        predy = xb + wxs
        self.predy = predy
        self.u = self.y - self.predy

        # compute variance from gradients
        if var_flag:
            vb = np.zeros((self.k,self.k))
            xtxi = la.inv(xtx)
            vb[:kk,:kk] = xtxi
            wax = nlmod(alpha,xw,w,transform,gradflag=1)
            waxtwax = wax.T @ wax
            try:
                waxtwaxi = la.inv(waxtwax)
            except:
                raise Exception("Singular variance matrix for nonlinear part")
            vb[-h:,-h:] = waxtwaxi
            self.vm = vb * self.sig2
        elif var_flag == 0:
            self.ihess = ssmin.hess_inv            
            self.vm = self.ihess * self.sig2
        





class NSLX(BaseNSLX):

    '''
    Estimation of the nonlinear SLX model - inverse distance
    power function and negative exponential distance function supported
    Includes output of all results.

    Parameters
    ----------
    y              : numpy.ndarray or pandas.Series
                     nx1 array for dependent variable
    x              : numpy.ndarray or pandas object
                     Two dimensional array with n rows and one column for each
                     independent (exogenous) variable, excluding the constant
    coords         : an n by 2 array or a selection of two columns from a data frame
    params         : a list of tuples containing the two parameters for the construction
                     of the distance weights and the transformation: 
                     (k,distance_upper_bound,transformation)
                     if the list consists of a single element, the same parameters are
                     applied to all transformations
                     default is [(10,np.inf,"exponential")] for 10 knn neighbors, variable
                     bandwidth and exponential transformation
                     (see make_wnslx in UTILS)
    distance_metric: metric for distance computations, either "Euclidean" (default) or "Arc"
                     (for decimal lat-lon degrees)
    leafsize       : parameter used to creat KDTree, default is 30
    slx_vars       : list with True,False for selection of X variables to which SLX should be applied
                     default is "All"
    var_flag       : flag for variance computation, default = 1 - uses analytical derivation,
                     = 0 - uses numerical approximation with inverse hessian
    conv_flag      : flag for convergence diagnostics, default = 0 for no diagnostics
                     = 1 - prints our minimize convergence summary
    verbose        : boolean for intermediate results in nonlinear optimization, default is False
    options        : options specific to scipy minimize, such as {"disp":True}
                     (see scipy minimize docs)                   
    vm             : boolean
                     if True, include variance-covariance matrix in summary
                     results
    name_y         : string
                     Name of dependent variable for use in output
    name_x         : list of strings
                     Names of independent variables for use in output
    name_coords    : list of strings
                     Names of coordinate variables used in distance matrix
    name_ds        : string
                     Name of dataset for use in output
    latex          : boolean
                     Specifies if summary is to be printed in latex format

    Attributes
    ----------
    output    : dataframe
                regression results pandas dataframe
    summary   : string
                Summary of regression results and diagnostics (note: use in
                conjunction with the print command)
    y         : n by 1 numpy array with dependent variable
    x         : n by k array with explanatory variables, includes constant
    xw        : n by h array with selected columns of X that will have the
                W(alpha) transformation applied to them
    w         : list of sparse CSR arrays to use in lag transformation, if
                list has a single element, same weights applied for all
    n         : number of observations
    k         : number of explanatory variables in X (includes constant)
    transform : tuple of transformations, either "power" or "exponential"
                when same transformation applies to all, tuple is a single
                element tuple (default is "power")
    verbose   : option for nonlinear optimization, either False (default) or
                True
    options   : options specific to scipy minimize, such as {"disp":True}
                (see scipy minimize docs)
    betas     : numpy array with parameter estimates
    utu       : sum of squared residuals
    ihess     : inverse of Hessian matrix
    sign      : estimate of residual variance (divided by n)
    sig2      : same as sign
    vm        : coefficient variance-covariance matrix (sign x ihess)
    predy     : vector of predicted values
    u         : vector of residuals
    ll        : float
                Log likelihood
    aic       : float
                Akaike information criterion
    schwarz   : float
                Schwarz information criterion    
    name_x    : variable names for explanatory variables
    name_ds   : data set name
    name_y    : name of dependent variable
    title     : output header
    
    Example
    --------
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> from libpysal.examples import load_example, get_path 
    >>> import spreg

    Open data on Chicago census tract SDOH variables from libpysal examples using geopandas.
    If you don't have chicagoSDOH installed, you can do so by running `load_example('chicagoSDOH')`. 

    >>> dfs = gpd.read_file(get_path('Chi-SDOH.shp'))

    For this example, we will use the 'HIS_ct' column (economic hardship index) as the 
    dependent variable and the 'Blk14P', 'Hisp14P', and 'EP_NOHSDP' columns as independent 
    variables. The coordinates "COORD_X" and "COORD_Y" will be used to construct the
    spatial weights matrix.

    >>> y = dfs[['HIS_ct']]
    >>> x = dfs[['Blk14P','Hisp14P','EP_NOHSDP']]
    >>> coords = dfs[["COORD_X","COORD_Y"]]

    For the regression, we set var_flag = 1 to obtain the analytical standard errors.

    >>> reg = spreg.NSLX(y, x, coords, var_flag=1)
    
    We can easily obtain a full summary of all the results nicely formatted and
    ready to be printed:

    >>> print(reg.summary)
    REGRESSION RESULTS
    ------------------
    <BLANKLINE>
    SUMMARY OF OUTPUT: NONLINEAR SLX
    --------------------------------
    Data set            :     unknown
    Dependent Variable  :      HIS_ct                Number of Observations:         791
    Mean dependent var  :     39.7301                Number of Variables   :           7
    S.D. dependent var  :     13.8098                Degrees of Freedom    :         784
    Sigma-square        :      32.287                Sum squared residual  :     25538.9
    S.E. of regression  :       5.682                Log likelihood        :   -2496.609
    Schwarz criterion   :    5039.931                Akaike info criterion :    5007.218
    Coordinates         : COORD_X, COORD_Y           Distance metric       :   Euclidean
    <BLANKLINE>
    ------------------------------------------------------------------------------------
                Variable     Coefficient       Std.Error     z-Statistic     Probability
    ------------------------------------------------------------------------------------
                CONSTANT        17.90865         0.43461        41.20635         0.00000
                  Blk14P         0.17910         0.00806        22.21475         0.00000
                 Hisp14P         0.05818         0.01525         3.81435         0.00014
               EP_NOHSDP         0.65462         0.02750        23.80062         0.00000
               We_Blk14P        17.16669         1.70331        10.07841         0.00000
              We_Hisp14P        88.30447     81337.66263         0.00109         0.99913
            We_EP_NOHSDP        10.02114         0.36436        27.50353         0.00000
    ------------------------------------------------------------------------------------
    Transformation: exponential
    KNN: 10
    Distance upper bound: inf
    ================================ END OF REPORT =====================================

    '''
    
    def __init__(
        self,
        y,
        x,
        coords,
        params = [(10,np.inf,"exponential")],
        distance_metric = "Euclidean",
        leafsize = 30,
        slx_vars="All",
        var_flag=1,
        conv_flag=0,
        verbose = False,
        options = None,
        vm = False,
        name_y=None,
        name_x=None,
        name_ds=None,
        name_coords=None,
        latex=False,
    ):
        
        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        
        coords,name_coords = USER.check_coords(coords,name_coords)
        x_constant, name_x, warn = USER.check_constant(x, name_x)
        
        name_x = USER.set_name_x(name_x, x_constant) # needs to be initialized for none, now with constant
        set_warn(self, warn)

        g = len(params)

        transform = [ i[2] for i in params ]
        
        xw = x_constant[:,1:]
        
        xw_name = name_x[1:] 
        if isinstance(slx_vars,list):   # not W(a)X for all X

            xw = xw[:,slx_vars]
            xw_name = list(compress(xw_name,slx_vars))

            if g > 1:

                if g == len(slx_vars):  # params does not have correct form
                    params = list(compress(params,slx_vars))
                    transform = list(compress(transform,slx_vars))
                elif g != len(xw_name):
                    raise Exception("Mismatch dimension params and slx_vars")

        if g == 1:
            cp = transform[0][0]
            xw_name = [ "W"+ cp + "_"+ i for i in xw_name]
        elif len(transform) > 1:
            if len(transform) != len(xw_name):
                raise Exception("Dimension mismatch")
            else:
                xw_name = ["W" + i[0][0] + "_" + j for i,j in zip(transform,xw_name)]

        self.name_x = name_x + xw_name

        # build up w list
        w = [0 for i in range(len(params))]
        for i in range(len(params)):
            w[i] = make_wnslx(coords,params[i],leafsize=leafsize,distance_metric=distance_metric)

        BaseNSLX.__init__(self,y=y,x=x_constant,xw=xw,w=w,transform=transform,
                          var_flag=var_flag,
                          verbose=verbose,options=options)
        
        if conv_flag:
            print("Convergence Characteristics")
            print("Success      :",self.success)
            print("Status       :",self.status)
            print("Message      :",self.message)
            print("Iterations   :",self.nit)
        
        self.ll = log_likelihood(self)
        self.aic = akaike(self)
        self.schwarz = schwarz(self)

        self.distance_metric = distance_metric
        self.knn = [ i[0] for i in params ]
        self.d_upper_bound = [ i[1] for i in params ]
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_coords = name_coords
        self.title = "NONLINEAR SLX"
        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['x'] * len(name_x) + ['wx'] * len(xw_name)
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top = _nslx_out(self, "top")
        self.other_mid = _nslx_out(self, "mid")
        output(reg=self, vm=vm, latex=latex)



def nslxobj(gamma0,y,x,xw,w,transform,verbose):
    '''
    Objective function for minimize call, computes the sum of squared residuals in
    the nonlinear SLX model. Note that all parameters other than gamma0 are passed
    in an arg parameter to the minimize function

    Parameters
    ----------
    gamma0      : current parameter estimates, consists of beta (for X) and alpha (for WX)
                  needs to be a flattened array
    y           : n by 1 vector with observations on the dependent variable
    x           : n by k matrix with observations on X, must include constant vector
    xw          : n by h matrix with columns of X that will be spatially lagged
    w           : list of sparse CSR weights for the X column transformations; if same weights
                  are used for all columns, must be a single element list
    transform   : list of transformations (either "power" or "exponential") to be applied to each
                  column of xw; if same transformation is applied to all, must be a single element
                  list
    verbose     : verbose option, whether or not the intermediate parameter values and residual sum
                  of squares are printed out

    Returns
    -------
    res2        : sum of squared residuals

    '''
    n = xw.shape[0]
    h = xw.shape[1]
    if verbose:
        print("gamma0",gamma0)
    b0 = gamma0[0:-h]
    alpha0 = gamma0[-h:]
    # create WX
    wx = nlmod(alpha0,xw,w,transform,gradflag=0)
    wxs = np.sum(wx,axis=1).reshape(-1,1)
    xb = x @ b0.reshape(-1,1)
    res = y - xb - wxs
    res2 = res.T @ res
    if verbose:
        print("res2",res2)

    return res2

def nlmod(alpha,xw,w,transform,gradflag=0):
    '''
    Constructs the matrix of spatially lagged X variables W(a)X (for gradflag = 0) and 
    the gradient matrix d(W(x))/d(a) (for gradflag = 1), for possibly different
    alpha parameters, different weights and different transformations (the transformations
    must match the weights and are not checked for compatibility). Calls nltransform for
    each relevant column of X. This allows the possibility that not all columns of X are
    used, defined by slx_vars.

    Parameters
    ----------
    alpha       : array with alpha parameters, same number as relevant columns in X
                  must be flattened (not a vector)
    xw          : matrix with relevant columns of X to be lagged
    w           : a list containing the weights matrix (as sparse CSR) for each column
                  if the same weights matrix are used for all columns, w is a single element list,
                  otherwise, the number of weights must match the number of relevant columns in X and
                  the number of elements in the transform tuple
    transform   : a tuple with the transformations for each relevant column in X, either "power" or "exponential"
                  if the same transformation is used for all columns, transform is a single element tuple
                  (transform,), otherwise, the number of elements in the tuple must match the number of relevant 
                  columns in X and the number of elements in the weights matrix tuple
                  the transformation must match the type of weights, but this is not checked
    gradflag    : flag for computation of gradient matrix, = 0 by default (actual function)
                  = 1 - computes matrix of first partial derivatives with respect to alpha

    Returns
    -------
    wx          : matrix with spatially lagged X variables

    '''
    # alpha must be flattened
    h = len(alpha)   # number of parameters
    if xw.shape[1] != h:
        raise Exception("Incompatible dimensions")
    g = len(w)       # number of weights
    if len(transform) != g:
        raise Exception("Incompatible dimensions")

    # initialize matrix of spatial lags
    n = xw.shape[0]
    wx = np.zeros((n,h))
    
    for i in range(h):
        
        if g == 1:
            walpha = _nltransform(alpha[i],w[0],transform[0],gradflag=gradflag)   # element of single element tuple
        elif g > 1:
            walpha = _nltransform(alpha[i],w[i],transform[i],gradflag=gradflag)    
        else:
            raise Exception("Operation not supported")
        wx[:,i] = walpha @ xw[:,i] 
    return wx


def _nltransform(a,w,transform,gradflag=0):
    '''
    Constructs the transformed CSR sparse array for power and exponential transformation
    for a given alpha parameter, input array and transformation. Note that the alpha parameters
    are positive, but are used as negative powers in the exponential transformation.

    Parameters
    ----------
    a           : alpha coefficient as a (positive) scalar
    w           : CSR sparse array with weights
    transform   : transformation, either "power" or "exponential"
    gradflag    : flag for gradient computation, 0 = function, default
                  1 = gradient

    Returns
    -------
    walpha      : CSR sparse array with transformed weights

    '''
    walpha = w.copy()

    if transform.lower() == "exponential":
        
        wdata = walpha
        awdata = wdata * a
        
        if gradflag == 0:
        
            awdata = -awdata
            np.exp(awdata.data, out= awdata.data)
            walpha = awdata

        elif gradflag == 1:

            ww = -awdata
            np.log(awdata.data, out = awdata.data)
            wln = awdata
            w1 = wln.multiply(wdata)
            np.exp(ww.data, out=ww.data)
            wgrad = ww.multiply(w1)
            wgrad = -wgrad
            walpha = wgrad

    elif transform.lower() == "power":

        if gradflag == 0:
            walpha = walpha.power(a)
        elif gradflag == 1:
            walpha = walpha.power(a)
            ww = w.copy()
            np.log(ww.data, out = ww.data)
            wgrad = walpha.multiply(ww)
            walpha = wgrad
            

    else:
        raise Exception("Transformation not supported")
    
    return walpha



def _test():
    import doctest

    # the following line could be used to define an alternative to the '<BLANKLINE>' flag
    # doctest.BLANKLINE_MARKER = 'something better than <BLANKLINE>'
    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == "__main__":
    _test()

    from libpysal.examples import load_example, get_path 
    import geopandas as gpd
    import spreg

    dfs = gpd.read_file(load_example('columbus').get_path("columbus.shp"))
    dfs['geometry'] = gpd.points_from_xy(dfs['X'], dfs['Y']) # Transforming polygons to points
    y = dfs[["CRIME"]]
    x = dfs[["INC","HOVAL","DISCBD"]]
    reg = spreg.NSLX(y, x, dfs['geometry'])
    print(reg.output)
    print(reg.summary)

