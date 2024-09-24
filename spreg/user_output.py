"""Internal helper files for user output."""

__author__ = (
    "Luc Anselin lanselin@gmail.com, "
    "Pedro Amaral pedrovma@gmail.com"
    "David C. Folch david.folch@asu.edu, "
    "Levi John Wolf levi.john.wolf@gmail.com, "
    "Jing Yao jingyao@asu.edu"
)
import numpy as np
import pandas as pd
import geopandas as gpd     # new for check_coords
import copy as COPY
from . import sputils as spu
from libpysal import weights
from libpysal import graph
from scipy.sparse.csr import csr_matrix
from .utils import get_lags        # new for flex_wx
from itertools import compress     # new for lfex_wx


def set_name_ds(name_ds):
    """Set the dataset name in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------

    name_ds     : string
                  User provided dataset name.

    Returns
    -------

    name_ds     : string

    """
    if not name_ds:
        name_ds = "unknown"
    return name_ds


def set_name_y(name_y):
    """Set the dataset name in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------
    name_ds     : string
                  User provided dataset name.

    Returns
    -------
    name_ds     : string

    """
    if not name_y:
        name_y = "dep_var"
    return name_y


def set_name_x(name_x, x, constant=False):
    """Set the independent variable names in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------
    name_x      : list of string
                  User provided exogenous variable names.

    x           : array
                  User provided exogenous variables including the constant.
    constant    : boolean
                  If False (default), constant name not included in name_x list yet
                  Append 'CONSTANT' at the front of the names

    Returns
    -------
    name_x      : list of strings

    """
    if not name_x:
        name_x = ["var_" + str(i + 1) for i in range(x.shape[1] - 1 + int(constant))]
    else:
        name_x = name_x[:]
    if not constant:
        name_x.insert(0, "CONSTANT")
    return name_x


def set_name_yend(name_yend, yend):
    """Set the endogenous variable names in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------
    name_yend   : list of strings
                  User provided exogenous variable names.

    Returns
    -------
    name_yend   : list of strings

    """
    if yend is not None:
        if not name_yend:
            return ["endogenous_" + str(i + 1) for i in range(len(yend[0]))]
        else:
            return name_yend[:]
    else:
        return []


def set_name_q(name_q, q):
    """Set the external instrument names in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------
    name_q      : string
                  User provided instrument names.
    q           : array
                  Array of instruments

    Returns
    -------
    name_q      : list of strings

    """
    if q is not None:
        if not name_q:
            return ["instrument_" + str(i + 1) for i in range(len(q[0]))]
        else:
            return name_q[:]
    else:
        return []


def set_name_yend_sp(name_y):
    """Set the spatial lag name in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------
    name_y      : string
                  User provided dependent variable name.

    Returns
    -------
    name_yend_sp : string

    """
    return "W_" + name_y

def set_name_spatial_lags(names, w_lags):
    """Set the spatial lag names for multiple variables and lag orders"

    Parameters
    ----------
    names      : string
                 Original variables' names.

    Returns
    -------
    lag_names : string

    """
    lag_names = ["W_" + s for s in names]
    for i in range(w_lags-1):
        lag_names += ["W" + str(i+2) + "_" + s for s in names]
    return lag_names

def set_name_q_sp(name_x, w_lags, name_q, lag_q, force_all=False):
    """Set the spatial instrument names in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------
    name_x      : list of strings
                  User provided exogenous variable names.
    w_lags      : int
                  User provided number of spatial instruments lags

    Returns
    -------
    name_q_sp   : list of strings

    """
    if force_all:
        names = name_x
    else:
        names = name_x[1:]  # drop the constant
    if lag_q:
        names = names + name_q
    sp_inst_names = []
    existing_names = set(names)
    name_count = {} # Dictionary to store the count of each name

    for name in names:
        if not name.startswith("W_"):
            name_count[name] = 2
        else:
            if name[2:] not in name_count:
                name_count[name[2:]] = 2

    for i in range(w_lags):
        for name in names:
            if name.startswith("W_"):
                lag_name = f"W{name_count[name[2:]]}_{name[2:]}"
                name_count[name[2:]] += 1
            else:
                if ("W_" + name) not in existing_names:
                    lag_name = f"W_{name}"
                else:
                    lag_name = f"W{name_count[name]}_{name}"
                    name_count[name] += 1
            sp_inst_names.append(lag_name)
            existing_names.add(lag_name)
    return sp_inst_names


def set_name_h(name_x, name_q):
    """Set the full instruments names in regression; return generic name if user
    provides no explicit name."

    Parameters
    ----------
    name_x      : list of strings
                  User provided exogenous variable names.
    name_q      : list of strings
                  User provided instrument variable names.

    Returns
    -------
    name_h      : list of strings

    """
    return name_x + name_q


def set_robust(robust):
    """Return generic name if user passes None to the robust parameter in a
    regression. Note: already verified that the name is valid in
    check_robust() if the user passed anything besides None to robust.

    Parameters
    ----------
    robust      : string or None
                  Object passed by the user to a regression class

    Returns
    -------
    robust      : string

    """
    if not robust:
        return "unadjusted"
    return robust


def set_name_w(name_w, w):
    """Return generic name if user passes None to the robust parameter in a
    regression. Note: already verified that the name is valid in
    check_robust() if the user passed anything besides None to robust.

    Parameters
    ----------
    name_w      : string
                  Name passed in by user. Default is None.
    w           : W object
                  pysal W object passed in by user

    Returns
    -------
    name_w      : string

    """
    if w != None:
        if name_w != None:
            return name_w
        else:
            return "unknown"
    return None


def set_name_multi(
    multireg,
    multi_set,
    name_multiID,
    y,
    x,
    name_y,
    name_x,
    name_ds,
    title,
    name_w,
    robust,
    endog=False,
    sp_lag=False,
):
    """Returns multiple regression objects with generic names

    Parameters
    ----------
    endog       : tuple
                  If the regression object contains endogenous variables, endog must have the
                  following parameters in the following order: (yend, q, name_yend, name_q)
    sp_lag       : tuple
                  If the regression object contains spatial lag, sp_lag must have the
                  following parameters in the following order: (w_lags, lag_q)

    """
    name_ds = set_name_ds(name_ds)
    name_y = set_name_y(name_y)
    name_x = set_name_x(name_x, x)
    name_multiID = set_name_ds(name_multiID)
    if endog or sp_lag:
        name_yend = set_name_yend(endog[2], endog[0])
        name_q = set_name_q(endog[3], endog[1])
    for r in multi_set:
        multireg[r].title = title + "%s" % r
        multireg[r].name_ds = name_ds
        multireg[r].robust = set_robust(robust)
        multireg[r].name_w = name_w
        multireg[r].name_y = "%s_%s" % (str(r), name_y)
        multireg[r].name_x = ["%s_%s" % (str(r), i) for i in name_x]
        multireg[r].name_multiID = name_multiID
        if endog or sp_lag:
            multireg[r].name_yend = ["%s_%s" % (str(r), i) for i in name_yend]
            multireg[r].name_q = ["%s_%s" % (str(r), i) for i in name_q]
            if sp_lag:
                multireg[r].name_yend.append(set_name_yend_sp(multireg[r].name_y))
                multireg[r].name_q.extend(
                    set_name_q_sp(
                        multireg[r].name_x, sp_lag[0], multireg[r].name_q, sp_lag[1]
                    )
                )
            multireg[r].name_z = multireg[r].name_x + multireg[r].name_yend
            multireg[r].name_h = multireg[r].name_x + multireg[r].name_q
    return multireg


def check_arrays(*arrays):
    """Check if the objects passed by a user to a regression class are
    correctly structured. If the user's data is correctly formed this function
    returns the number of observations, if not then an exception is raised. Note, this does not
    check for model setup, simply the shape and types of the objects.

    Parameters
    ----------
    *arrays : anything
              Objects passed by the user to a regression class; any type
              object can be passed and any number of objects can be passed

    Returns
    -------
    Returns : int
              number of observations

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import check_arrays
    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> # Extract CRIME column from the dbf file
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> n = check_arrays(y, X)
    >>> print(n)
    49

    """
    rows = []
    for i in arrays:
        if i is None:
            continue
        if isinstance(i, (pd.Series, pd.DataFrame)):
            i = i.to_numpy()
        if not isinstance(i, (np.ndarray, csr_matrix)):
            raise Exception(
                "all input data must be either numpy arrays or sparse csr matrices"
            )
        shape = i.shape
        if len(shape) > 2:
            raise Exception("all input arrays must have two dimensions")
        if len(shape) == 1:
            shape = (shape[0], 1)
        if shape[0] < shape[1]:
            raise Exception("one or more input arrays have more columns than rows")
        if not spu.spisfinite(i):
            raise Exception("one or more input arrays have missing/NaN values")
        rows.append(shape[0])
    if len(set(rows)) > 1:
        raise Exception("arrays not all of same length")
    return rows[0]


def check_y(y, n, name_y=None):
    """Check if the y object passed by a user to a regression class is
    correctly structured. If the user's data is correctly formed this function
    returns nothing, if not then an exception is raised. Note, this does not
    check for model setup, simply the shape and types of the objects.

    Parameters
    ----------
    y       : anything
              Object passed by the user to a regression class; any type
              object can be passed

    n       : int
              number of observations
    
    name_y  : string
              Name of the y variable

    Returns
    -------
    y       : anything
              Object passed by the user to a regression class
    
    name_y  : string
              Name of the y variable

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import check_y
    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')

    # Extract CRIME column from the dbf file

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> y = check_y(y, 49)

    # should not raise an exception

    """
    if isinstance(y, (pd.Series, pd.DataFrame)):
        if not name_y:
            try:
                name_y = y.name
            except AttributeError:
                name_y = y.columns.to_list()
                if len(name_y) == 1:
                    name_y = name_y[0]
                    
        y = y.to_numpy()
    if not isinstance(y, np.ndarray):
        print(y.__class__.__name__)
        raise Exception("y must be a numpy array or a pandas Series")
    shape = y.shape
    if len(shape) > 2:
        raise Exception("all input arrays must have two dimensions")
    if len(shape) == 1:
        try:
            y = y.reshape(n, 1)
        except:
            raise Exception(
                "y must be a single column array matching the length of other arrays"
            )
    if y.shape != (n, 1):
        raise Exception(
            "y must be a single column array matching the length of other arrays"
        )
    return y, name_y

def check_endog(arrays, names):
    """Check if each of the endogenous arrays passed by a user to a regression class are
    pandas objects. In this case, the function converts them to numpy arrays and collects their names.

    Parameters
    ----------
    arrays : list
              List of endogenous variables passed by the user to a regression class
    names  : list
              List of names of the endogenous variables, assumed in the same order as the arrays
    """
    for i in range(len(arrays)):
        if isinstance(arrays[i], (pd.Series, pd.DataFrame)):
            if not names[i]:
                try:
                    names[i] = [arrays[i].name]
                except AttributeError:
                    names[i] = arrays[i].columns.to_list()
            arrays[i] = arrays[i].to_numpy()

        if arrays[i] is None:
            pass
        elif len(arrays[i].shape) == 1:
            arrays[i].shape = (arrays[i].shape[0], 1)
    return (*arrays, *names)

def check_weights(w, y, w_required=False, time=False, slx_lags=0, allow_wk=False):
    """Check if the w parameter passed by the user is a libpysal.W object and
    check that its dimensionality matches the y parameter.  Note that this
    check is not performed if w set to None.

    Parameters
    ----------
    w       : any python object
              Object passed by the user to a regression class; any type
              object can be passed
    y       : numpy array
              Any shape numpy array can be passed. Note: if y passed
              check_arrays, then it will be valid for this function
    w_required : boolean
                 True if a W matrix is required, False (default) if not.
    time    : boolean
              True if data contains a time dimension.
              False (default) if not.
    slx_lags : int
                Number of lags of X in the spatial lag model.
    allow_wk : boolean
                True if Kernel weights are allowed as W, False (default) if not.

    Returns
    -------
    Returns : w
              weights object passed by the user to a regression class

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import check_weights
    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> # Extract CRIME column from the dbf file
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> w = libpysal.io.open(libpysal.examples.get_path("columbus.gal"), 'r').read()
    >>> w = check_weights(w, y)

    # should not raise an exception

    """
    if w_required == True or (w is not None) or slx_lags > 0:
        if isinstance(w, graph.Graph):
            w = w.to_W()
            
        if w == None:
            raise Exception("A weights matrix w must be provided to run this method.")
        
        if not isinstance(w, weights.W):
            from warnings import warn
            warn("w must be API-compatible pysal weights object")

        # check for kernel weights, if so insert zeros on diagonal
        if slx_lags == 1 and isinstance(w, weights.Kernel) and allow_wk:
            w = weights.fill_diagonal(w,val=0.0)
        elif slx_lags > 1 and isinstance(w, weights.Kernel):
            raise Exception("Higher orders not supported for kernel weights")
        elif isinstance(w, weights.Kernel) and (slx_lags == 0 or not allow_wk):
            raise Exception("Kernel weights not allowed as W for this model")

        if w.n != y.shape[0] and time == False:
            raise Exception("y must have n rows, and w must be an nxn PySAL W object")
        diag = w.sparse.diagonal()
        # check to make sure all entries equal 0
        if diag.min() != 0 or diag.max() != 0:
            raise Exception("All entries on diagonal must equal 0.")
        
    return w


def check_robust(robust, wk):
    """Check if the combination of robust and wk parameters passed by the user
    are valid. Note: this does not check if the W object is a valid adaptive
    kernel weights matrix needed for the HAC.

    Parameters
    ----------
    robust  : string or None
              Object passed by the user to a regression class
    w       : any python object
              Object passed by the user to a regression class; any type
              object can be passed

    Returns
    -------
    Returns : nothing
              Nothing is returned

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import check_robust
    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> # Extract CRIME column from the dbf file
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> wk = None
    >>> check_robust('White', wk)

    # should not raise an exception

    """
    if robust:
        if robust.lower() == "hac":
            if not isinstance(wk, weights.Kernel):
                raise Exception("HAC requires that wk be a Kernel Weights object")
            diag = wk.sparse.diagonal()
            # check to make sure all entries equal 1
            if diag.min() < 1.0:
                print(diag.min())
                raise Exception(
                    "All entries on diagonal of kernel weights matrix must equal 1."
                )
            if diag.max() > 1.0:
                print(diag.max())
                raise Exception(
                    "All entries on diagonal of kernel weights matrix must equal 1."
                )
            # ensure off-diagonal entries are in the set of real numbers [0,1)
            wegt = wk.weights
            for i in wk.id_order:
                vals = wegt[i]
                vmin = min(vals)
                vmax = max(vals)
                if vmin < 0.0:
                    raise Exception(
                        "Off-diagonal entries must be greater than or equal to 0."
                    )
                if vmax > 1.0:
                    # NOTE: we are not checking for the case of exactly 1.0 ###
                    raise Exception("Off-diagonal entries must be less than 1.")
        elif robust.lower() == "white" or robust.lower() == "ogmm":
  #          if wk:
  #              raise Exception("White requires that wk be set to None")
            pass      # these options are not affected by wk
        else:
            raise Exception(
                "invalid value passed to robust, see docs for valid options"
            )

''' Deprecated in 1.6.1
def check_spat_diag(spat_diag, w):
    """Check if there is a w parameter passed by the user if the user also
    requests spatial diagnostics.

    Parameters
    ----------
    spat_diag   : boolean
                  Value passed by a used to a regression class
    w           : any python object
                  Object passed by the user to a regression class; any type
                  object can be passed

    Returns
    -------
    Returns : nothing
              Nothing is returned

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import check_spat_diag
    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> # Extract CRIME column from the dbf file
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> w = libpysal.io.open(libpysal.examples.get_path("columbus.gal"), 'r').read()
    >>> check_spat_diag(True, w)

    # should not raise an exception

    """
    if spat_diag:
        if not isinstance(w, weights.W):
            raise Exception("w must be a libpysal.W object to run spatial diagnostics")
'''


def check_spat_diag(spat_diag, w, robust=None, slx_lags=None):
    """Check if spatial diagnostics should be disabled given argument combinations.

    Parameters
    ----------
    spat_diag   : boolean
                  Value passed by a used to a regression class
    w           : weights object
    robust      : string
                  variance-covariance estimator


    Returns
    -------
    spat_diag   : boolean
                  Updated spatial diagnostics flag
    warn        : string
                 Warning message if it is not possible to run spatial diagnostics

    """
    warn = None
    if robust == "hac" and spat_diag:
            warn = "Spatial diagnostics are not available for HAC estimation.\nHence, spatial diagnostics have been disabled for this model."
            spat_diag = False

    if spat_diag and isinstance(w, weights.Kernel) and slx_lags > 0:
            warn = "\nSpatial diagnostics are not currently available for SLX models with kernel weights.\nHence, spatial diagnostics have been disabled for this model."
            spat_diag = False

    return spat_diag, warn

def check_reg_list(regimes, name_regimes, n):
    """Check if the regimes parameter passed by the user is a valid list of
    regimes. Note: this does not check if the regimes are valid for the
    regression model.

    Parameters
    ----------
    regimes     : list or np.array or pd.Series
                    Object passed by the user to a regression class
    name_regimes : string
                    Name of the regimes variable    
    n           : int
                    number of observations

    Returns
    -------
    regimes     : list
                    regimes object passed by the user to a regression class as a list
    name_regimes : string

    """
    if isinstance(regimes, list):
        pass
    elif isinstance(regimes, pd.Series):
        if not name_regimes:
            name_regimes = regimes.name
        regimes = regimes.tolist()
    elif isinstance(regimes, np.ndarray):
        regimes = regimes.tolist()
    else:
        raise Exception("regimes must be a list, numpy array, or pandas Series")
    if len(regimes) != n:
        raise Exception(
            "regimes must have the same number of observations as the dependent variable"
        )
    return regimes, name_regimes







def check_regimes(reg_set, N=None, K=None):
    """Check if there are at least two regimes

    Parameters
    ----------
    reg_set     : list
                  List of the regimes IDs

    Returns
    -------
    Returns : nothing
              Nothing is returned

    """
    if len(reg_set) < 2:
        raise Exception(
            "At least 2 regimes are needed to run regimes methods. Please check your regimes variable."
        )
    if 1.0 * N / len(reg_set) < K + 1:
        raise Exception(
            "There aren't enough observations for the given number of regimes and variables. Please check your regimes variable."
        )


def check_constant(x, name_x=None, just_rem=False):
    """Check if the X matrix contains a constant. If it does, drop the constant and replace by a vector of ones.

    Parameters
    ----------
    x           : array
                  Value passed by a used to a regression class
    name_x      : list of strings
                  Names of independent variables
    just_rem    : boolean
                  If False (default), remove all constants and add a vector of ones
                  If True, just remove all constants
    Returns
    -------
    x_constant : array
                 Matrix with independent variables plus constant
    name_x     : list of strings
                 Names of independent variables (updated if any variable droped)
    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import check_constant
    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> x_constant,name_x,warn = check_constant(X)
    >>> x_constant.shape
    (49, 3)

    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        if name_x is None:
            try:
                name_x = x.columns.to_list()
            except AttributeError:
                name_x = x.name
        x = x.to_numpy()
    x_constant = COPY.copy(x)
    keep_x = COPY.copy(name_x)
    warn = None
    if isinstance(x_constant, np.ndarray):
        diffs = np.ptp(x_constant, axis=0)
        if sum(diffs == 0) > 0:
            x_constant = np.delete(x_constant, np.nonzero(diffs == 0), 1)
    else:
        diffs = (x_constant.max(axis=0).toarray() - x_constant.min(axis=0).toarray())[0]
        if sum(diffs == 0) > 0:
            x_constant = x_constant[:, np.nonzero(diffs > 0)[0]]

    if sum(diffs == 0) > 0:
        if keep_x:
            rem_x = [keep_x[i] for i in np.nonzero(diffs == 0)[0]]
            warn = "Variable(s) " + str(rem_x) + " removed for being constant."
            keep_x[:] = [keep_x[i] for i in np.nonzero(diffs > 0)[0]]
        else:
            if sum(diffs == 0) == 1:
                warn = "One variable has been removed for being constant."
            else:
                warn = (
                    str(sum(diffs == 0))
                    + " variables have been removed for being constant."
                )
    if not just_rem:
        return spu.sphstack(np.ones((x_constant.shape[0], 1)), x_constant), keep_x, warn
    else:
        return x_constant, keep_x, warn


def flex_wx(w,x,name_x,constant=True,slx_lags=1,slx_vars="All"):
    """
    Adds spatially lagged variables to an existing x matrix with or without a constant term
    Adds variable names prefaced by W_ for the lagged variables
    Allows flexible selection of x-variables to be lagged through list of booleans

    Arguments
    ---------
    w        : PySAL supported spatial weights
    x        : input matrix of x variables
    name_x   : input list of variable names for x
    constant : flag for whether constant is included in x, default = True
               no spatial lags are computed for constant term
    slx_lags : order of spatial lags, default = 1
    slx_vars : either "All" (default) for all variables lagged, or a list
               of booleans matching the columns of x that will be lagged or not

    Returns
    -------
    a tuple with
    bigx     : concatenation of original x matrix and spatial lags
    bignamex : list of variable names including spatial lags

    """
    if constant == True:
        xwork = x[:,1:]
        xnamework = name_x[1:]        
    else:
        xwork = x
        xnamework = name_x
    
    if isinstance(slx_vars,list):
        if len(slx_vars) == len(xnamework):
            xwork = xwork[:,slx_vars]
            xnamework = list(compress(xnamework,slx_vars))
        else:
            raise Exception("Mismatch number of columns and length slx_vars")
    
    lagx = get_lags(w,xwork,slx_lags)
    xlagname = set_name_spatial_lags(xnamework,slx_lags)
    bigx = np.hstack((x,lagx))
    bignamex = name_x + xlagname
    return(bigx,bignamex)

def check_coords(data,name_coords):
    '''
    Checks to make sure the object passed is turned into a numpy array of coordinates.

    Parameters
    ----------
    data        : an n by 2 array or a selection of two columns from a data frame

    Returns
    -------
    xy          : n by 2 numpy array with coordinates
    name_coords : names for coordinate variables

    '''
    if (not(isinstance(data,np.ndarray))):
        if (isinstance(data,pd.core.frame.DataFrame)):
            xy = np.array(data)

            if not name_coords:
                try:
                    name_coords = data.name
                except AttributeError:
                    name_coords = data.columns.to_list()

        elif (isinstance(data,gpd.geoseries.GeoSeries)):   # geometry column
            if data[0].geom_type == 'Point':
                xy = data.get_coordinates()
                xy = np.array(xy)

                if not name_coords:
                    try:
                        name_coords = data.name
                    except AttributeError:
                        name_coords = data.columns.to_list()

            else:
                raise Exception("Data type not supported")
        else:
            raise Exception("Data type not supported")
    else:
        xy = data
        if not name_coords:
            name_coords = "unknown"
    if xy.shape[1] != 2:
        raise Exception("Incompatible dimensions")
    return xy,name_coords

def _test():
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    _test()
