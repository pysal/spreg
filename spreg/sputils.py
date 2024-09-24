import numpy as np
import numpy.linalg as la
import scipy.sparse as SP
import pandas as pd
import geopandas as gpd
from scipy.sparse import linalg as SPla
from itertools import compress
import libpysal.weights as weights
from libpysal import graph
from scipy import sparse


def spdot(a, b, array_out=True):
    """
    Matrix multiplication function to deal with sparse and dense objects

    Parameters
    ----------
    a           : array
                  first multiplication factor. Can either be sparse or dense.
    b           : array
                  second multiplication factor. Can either be sparse or dense.
    array_out   : boolean
                  If True (default) the output object is always a np.array

    Returns
    -------
    ab : array
         product of a times b. Sparse if a and b are sparse. Dense otherwise.
    """
    if type(a).__name__ == "ndarray" and type(b).__name__ == "ndarray":
        ab = np.dot(a, b)
    elif (
        type(a).__name__ == "csr_matrix"
        or type(b).__name__ == "csr_matrix"
        or type(a).__name__ == "csc_matrix"
        or type(b).__name__ == "csc_matrix"
    ):
        ab = a * b
        if array_out:
            if type(ab).__name__ == "csc_matrix" or type(ab).__name__ == "csr_matrix":
                ab = ab.toarray()
    else:
        raise Exception(
            "Invalid format for 'spdot' argument: %s and %s"
            % (type(a).__name__, type(b).__name__)
        )
    return ab


def spmultiply(a, b, array_out=True):
    """
    Element-wise multiplication function to deal with sparse and dense
    objects. Both objects must be of the same type.

    Parameters
    ----------
    a           : array
                  first multiplication factor. Can either be sparse or dense.
    b           : array
                  second multiplication factor. Can either be sparse or dense.
                  integer.
    array_out   : boolean
                  If True (default) the output object is always a np.array

    Returns
    -------
    ab : array
         elementwise multiplied object. Sparse if a is sparse. Dense otherwise.
    """

    if type(a).__name__ == "ndarray" and type(b).__name__ == "ndarray":
        ab = a * b
    elif (type(a).__name__ == "csr_matrix" or type(a).__name__ == "csc_matrix") and (
        type(b).__name__ == "csr_matrix" or type(b).__name__ == "csc_matrix"
    ):
        ab = a.multiply(b)
        if array_out:
            if type(ab).__name__ == "csc_matrix" or type(ab).__name__ == "csr_matrix":
                ab = ab.toarray()
    else:
        raise Exception(
            "Invalid format for 'spmultiply' argument: %s and %s"
            % (type(a).__name__, type(b).__name__)
        )
    return ab


def sphstack(a, b, array_out=False):
    """
    Horizontal stacking of vectors (or matrices) to deal with sparse and dense objects

    Parameters
    ----------
    a           : array or sparse matrix
                  First object.
    b           : array or sparse matrix
                  Object to be stacked next to a
    array_out   : boolean
                  If True the output object is a np.array; if False (default)
                  the output object is an np.array if both inputs are
                  arrays or CSR matrix if at least one input is a CSR matrix

    Returns
    -------
    ab          : array or sparse matrix
                  Horizontally stacked objects
    """
    if type(a).__name__ == "ndarray" and type(b).__name__ == "ndarray":
        ab = np.hstack((a, b))
    elif type(a).__name__ == "csr_matrix" or type(b).__name__ == "csr_matrix":
        ab = SP.hstack((a, b), format="csr")
        if array_out:
            if type(ab).__name__ == "csr_matrix":
                ab = ab.toarray()
    else:
        raise Exception(
            "Invalid format for 'sphstack' argument: %s and %s"
            % (type(a).__name__, type(b).__name__)
        )
    return ab


def spbroadcast(a, b, array_out=False):
    """
    Element-wise multiplication of a matrix and vector to deal with sparse
    and dense objects

    Parameters
    ----------
    a           : array or sparse matrix
                  Object with one or more columns.
    b           : array
                  Object with only one column
    array_out   : boolean
                  If True the output object is a np.array; if False (default)
                  the output object is an np.array if both inputs are
                  arrays or CSR matrix if at least one input is a CSR matrix

    Returns
    -------
    ab          : array or sparse matrix
                  Element-wise multiplication of a and b
    """
    if type(a).__name__ == "ndarray" and type(b).__name__ == "ndarray":
        ab = a * b
    elif type(a).__name__ == "csr_matrix":
        b_mod = SP.lil_matrix((b.shape[0], b.shape[0]))
        b_mod.setdiag(b)
        ab = (a.T * b_mod).T
        if array_out:
            if type(ab).__name__ == "csr_matrix":
                ab = ab.toarray()
    else:
        raise Exception(
            "Invalid format for 'spbroadcast' argument: %s and %s"
            % (type(a).__name__, type(b).__name__)
        )
    return ab


def spmin(a):
    """
    Minimum value in a matrix or vector to deal with sparse and dense objects

    Parameters
    ----------
    a           : array or sparse matrix
                  Object with one or more columns.

    Returns
    -------
    min a       : int or float
                  minimum value in a
    """
    return a.min()


def spmax(a):
    """
    Maximum value in a matrix or vector to deal with sparse and dense objects

    Parameters
    ----------
    a           : array or sparse matrix
                  Object with one or more columns.

    Returns
    -------
    max a       : int or float
                  maximum value in a
    """
    return a.max()


def splogdet(a):
    """
    Compute the log determinant of a large matrix.

    Parameters
    ----------
    a       :   array or sparse matrix
                Object with one or more columns

    Returns
    -------
    log determinant of a    :   int or float
                                logged determinant of a
    """
    if SP.issparse(a):
        LU = SPla.splu(a)
        det = np.sum(np.log(np.abs(LU.U.diagonal())))
    else:
        sgn, ldet = la.slogdet(a)
        det = sgn * ldet
    return det


def spfill_diagonal(a, val):
    """
    Fill the diagonal of a sparse or dense matrix
    Parameters
    ----------

    a       :   array or sparse matrix
                Object with one or more columns
    val     :   int or float
                value with which to fill the diagonal of a

    Returns
    -------
    a       :   array or sparse matrix
                with val on each element of the diagonal
    """
    if SP.issparse(a):
        a.setdiag(val)
    else:
        np.fill_diagonal(a, val)
    return a


def spinv(a):
    """
    Compute the inverse of a sparse or dense matrix

    Parameters
    ----------
    a       :   array or sparse matrix
                Object with one or more columns

    Returns
    -------
    ai      :   array or sparse matrix
                the inverse of a
    """
    if SP.issparse(a):
        ai = SPla.inv(a)
    else:
        ai = la.inv(a)
    return ai


def spisfinite(a):
    """
    Determine whether an array has nan or inf values

    Parameters
    ----------
    a   :   array or sparse matrix
            Object with one or more columns

    Returns
    -------
        :   bool
            denoting whether or not the array contains any NaN or inf
    """
    return np.isfinite(a.sum())


def spmultiplier(w, rho, method="simple", mtol=0.00000001):
    """"
    Spatial Lag Multiplier Calculation
    Follows Kim, Phipps and Anselin (2003) (simple), and LeSage and Pace (2009) (full, power)

    Attributes
    ----------
    w          : PySAL format spatial weights matrix
    rho        : spatial autoregressive coefficient
    method     : one of "simple" (default), "full" or "power"
    mtol       : tolerance for power iteration (default=0.00000001)

    Returns
    -------
    multipliers : dictionary with
                  ati = average total impact multiplier
                  adi = average direct impact multiplier
                  aii = average indirect impact multiplier
                  pow = powers used in power approximation (otherwise 0)

    """
    multipliers = {"ati": 1.0, "adi": 1.0, "aii": 1.0, "method": method, "warn": ''}
    multipliers["pow"] = 0
    multipliers["ati"] = 1.0 / (1.0 - rho)
    n = w.n
    if method == "simple":
        pass
    elif method == "full":
        wf = w.full()[0]
        id0 = np.identity(n)
        irw0 = (id0 - rho * wf)
        invirw0 = np.linalg.inv(irw0)
        adii0 = np.sum(np.diag(invirw0))
        multipliers["adi"] = adii0 / n
    elif method == "power":
        ws3 = w.to_sparse(fmt='csr')        
        rhop = rho
        ww = ws3
        pow = 1
        adi = 1.0
        adidiff = 100.00

        while adidiff > mtol:
            pow = pow + 1
            ww = SP.csr_matrix.dot(ww, ws3)
            trw = ww.diagonal().sum()
            rhop = rhop * rho
            adidiff = rhop * trw / n
            adi = adi + adidiff
        multipliers["adi"] = adi.item()
        multipliers["pow"] = pow
    else:
        multipliers["warn"] = "Method '"+method+"' not supported for spatial impacts.\n"
        multipliers["method"] ='simple'
    multipliers["aii"] = multipliers["ati"] - multipliers["adi"]
    return (multipliers)

def _sp_effects(reg, variables, spmult, slx_lags=0,slx_vars="All"):
    """
    Calculate spatial lag, direct and indirect effects
    
    Attributes
    ----------
    reg        : regression object
    variables  : chunk of self.output with variables to calculate effects
    spmult     : dictionary with spatial multipliers
    slx_lags   : number of SLX lags
    slx_vars   : either "All" (default) for all variables lagged, or a list
                 of booleans matching the columns of x that will be lagged or not

    Returns
    -------
    btot       : total effects
    bdir       : direct effects
    bind       : indirect effects
    """
    
    variables_x_index = variables.index

    m1 = spmult['ati']
    btot = m1 * reg.betas[variables_x_index]
    m2 = spmult['adi']
    bdir = m2 * reg.betas[variables_x_index]

    # Assumes all SLX effects are indirect effects. 
    if slx_lags > 0:
        if reg.output.regime.nunique() > 1:
            btot_idx = pd.Series(btot.flatten(), index=variables_x_index)   
            wchunk_size = len(variables.query("regime == @reg.output.regime.iloc[0]")) #Number of exogenous variables in each regime
            for i in range(slx_lags):
                chunk_indices = variables_x_index + (i+1) * wchunk_size
                bmult = m1 * reg.betas[chunk_indices]
                btot_idx[variables_x_index] += bmult.flatten()
                btot = btot_idx.to_numpy().reshape(btot.shape)

        else:
            variables_wx = reg.output.query("var_type == 'wx'")
            variables_wx_index = variables_wx.index
            if hasattr(reg, 'slx_vars') and isinstance(slx_vars,list):
                flexwx_indices = list(compress(variables_x_index,slx_vars))   # indices of x variables in wx
            else:
                flexwx_indices = variables_x_index    # all x variables
            xind = [h - 1 for h in flexwx_indices]
            wchunk_size = len(variables_wx_index)//slx_lags
            for i in range(slx_lags):
                start_idx = i * wchunk_size
                end_idx = start_idx + wchunk_size
                chunk_indices = variables_wx_index[start_idx:end_idx]
                bmult = m1 * reg.betas[chunk_indices]
                btot[xind] = btot[xind] + bmult

        bind = btot - bdir
    else:
        m3 = spmult['aii']
        bind = m3 * reg.betas[variables_x_index]

    return btot, bdir, bind

def i_multipliers(w,coef=0.0,model='lag',id=None):
    '''
    Creates pandas DataFrame with spatial multipliers with direct effects
    (diagonal), effect of neighbors (row sum) and effect on neighbors
    (column sum) for each observation

    Parameters
    ----------
    w         : spatial weights, either sparse CSR, PySAL weights object
                or full array
    coef      : spatial coefficient, default is 0.0; non-zero values for
                spatial autoregressive coefficient or parameter in nonlinear
                SLX -- coef has no effect on slx and kernel models
    model     : type of spatial model, default is 'lag' for spatial lag or 
                spatial Durbin model; other options are "slx" (contiguity
                weights without parameters), "kernel" (kernel weights),
                "power" (nonlinear SLX power model) and "exponential" (nonlinear
                SLX exponential model)
    id        : pandas Series with ID variable for each observation, default is none,
                which creates a vector with sequence numbers and assigns variable name
                "ID"; otherwise variable name is extracted from Series columns.

    '''
    
    if sparse.issparse(w):   # sparse kernel or dist fraction
        n = w.shape[0]
        edirect = np.zeros((n,1))
        ww = w.copy()
        if model == 'power':
            ww = ww.power(coef)
        elif model == 'exponential':

            wdata = ww.data    # uses data attribute of CSR to apply transformation
            wexp = np.exp(wdata * -coef)
            ww.data = wexp
        else:
            raise Exception("Model not supported")
        
    else:     # pysal weights
        if isinstance(w,weights.W):
            n = w.n
            wf = w.full()[0]
        elif isinstance(w, graph.Graph):
            w = w.to_W()
            n = w.n
            wf = w.full()[0]
        elif type(w) == np.ndarray:
            n = w.shape[0]  
            wf = w
        else:
            raise Exception("Weights object not supported")
        
        edirect = np.zeros((n,1))

        if model == 'slx':
            ww = wf
        elif model == 'kernel':
            edirect = np.diag(wf).copy().reshape(-1,1)
            ww = wf.copy()
            np.fill_diagonal(ww,0)                  
        elif model == 'lag':
            
            id0 = np.identity(n)
            irw0 = (id0 - coef * wf)
            invirw0 = np.linalg.inv(irw0)
            edirect = np.diag(invirw0).copy().reshape(-1,1)
            ww = invirw0.copy()
            np.fill_diagonal(ww,0)
        else:
            raise Exception("Model not supported")
            
    eofNbrs = ww.sum(axis=1).reshape(-1,1)
    eonNbrs = ww.sum(axis=0).reshape(-1,1) 

    if (isinstance(id,pd.core.frame.DataFrame)) or (isinstance(id,gpd.geoseries.GeoSeries)):
        pdid = id   # already a data frame
    elif isinstance(id,np.ndarray):
        pdid = pd.DataFrame(id,columns=["ID"])
    elif id == None:
        id = np.arange(n,dtype='int32').reshape(-1,1)
        pdid = pd.DataFrame(id,columns=["ID"])
    else:
        raise Exception("ID format not supported")
        
    impacts = np.concatenate((edirect,eofNbrs,eonNbrs),axis=1)  
    dfimpacts = pd.DataFrame(impacts,columns=["Direct","EofNbrs","EonNbrs"])
    dfimpacts = pd.concat([pdid,dfimpacts],axis=1)

    return dfimpacts
    

def _test():
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    _test()
