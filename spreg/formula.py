__author__ = "Tyler D. Hoffman pysal@tdhoffman.com"

import numpy as np
import pandas as pd
import geopandas as gpd
from formulaic import model_matrix
from .ols import OLS
from .ml_lag import ML_Lag
from .ml_error import ML_Error
from .twosls_sp import GM_Lag
from .error_sp import GM_Error, GM_Endog_Error, GM_Combo
from .error_sp_hom import GM_Error_Hom, GM_Endog_Error_Hom, GM_Combo_Hom
from .error_sp_het import GM_Error_Het, GM_Endog_Error_Het, GM_Combo_Het


def noninplace_remove(lst, el):
    """
    Helper function to remove an element from a list and return the list.
    """
    
    lst.remove(el)
    return lst


def expand_lag(w_name, fields):
    """
    Helper function for `from_formula`.
    Creates a formula string with spatial lag terms for the given fields.
    """

    fields = [fields] if type(fields) == str else fields
    output = ["{" + w_name + ".sparse @ `" + field + "`}" for field in fields]
    return " + ".join(output)


def from_formula(formula, df, w=None, method="gm", skedastic=None,
                 debug=False, **kwargs):
    """
    Given a formula and a dataframe, parse the formula and return a configured
    `spreg` model.
    
    Inputs
    ------
    
    formula  : string
               formula description following formulaic's grammar and the below syntax
    df       : pandas.DataFrame or geopandas.GeoDataFrame
               container with columns labelled according to the desired variables
    w        : libpysal.weights.W
               spatial weights matrix for spatial models. Default set to None.
    method   : string
               estimation method: "gm" for generalized method of moments,
               "full" for brute force ML estimation, "lu" for ML estimation with
               LU log Jacobian calculation, or "ord" for ML estimation with Ord
               log Jacobian calculation. Default set to "gm".
    skedastic: string
               specifies form of skedasticity in an error term. "het" for 
               heteroskedastic spatial error, "hom" for homoskedastic spatial error.
               Default set to None (regular spatial error estimation).
    debug    : boolean
               if true, outputs the preprocessed formula alongside the spatial model.
               Default set to False.
    **kwargs : dictionary
               additional keyword arguments to be passed to the underlying model class.


    Description
    -----------
    Syntax for the formulas is the same as `formulaic`, with two spatial operators added.

    - The `<...>` operator:
        - Enclose variables (covariates or response) in angle brackets to 
          denote a spatial lag.
        - `<` and `>` are not reserved characters in `formulaic` (the underlying 
          formula parsing library), so there are no conflicts.
        - **Usage:** include `<FIELD>` as a term in a formula string to add 
          that field and its spatial lag field from the dataframe to model matrix.
        - Can use other transformations within `<...>`, e.g. 
          `<{10*FIELD1} + FIELD2>` will be correctly parsed.
    - The `&` symbol:
        - Adds a spatial error component to a model.
        - `&` is not a reserved character in formulaic, so there are no conflicts.
        - **Usage:** include `&` as a term in a formula string to introduce 
          a spatial error component in a model.

    The parser accepts combinations of `<...>` and `&`: `<FIELD1 + ... + FIELDN> + &` 
    is the most general possible spatial model available. Additionally, any 
    transformations from [`formulaic`'s grammar](https://matthewwardrop.github.io/formulaic/concepts/grammar/) are admissible anywhere in the formula string.

    Variable names do not need to be provided separately to the model classes 
    (i.e., through the `name_x` and `name_y` keyword arguments) as they can be 
    automatically detected from the formula string. Variables which read 
    ``w.sparse @ `FIELD` `` are the spatial lags of those fields 
    (future TODO: make this prettier).

    Finally, if `yend` is included as a keyword argument for an error model,
    the dispatcher will send it to the correct endogenous error model.
    

    Examples
    --------

    Import necessary libraries.

    >>> import spreg
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> from libpysal.examples import load_example
    >>> from libpysal.weights import Kernel, fill_diagonal
    
    Load the Boston housing example.

    >>> boston = load_example("Bostonhsg")
    >>> boston_df = gpd.read_file(boston.get_path("boston.shp"))
    
    Create a weights matrix and set its diagonal to zero (necessary for spatial
    lag of y model).

    >>> weights = Kernel(boston_df[["x", "y"]], k=50, fixed=False)
    >>> weights = fill_diagonal(weights, 0)
    
    Write a formula. This formula incorporates transformations, a spatial lag of
    a covariate, a spatial lag of the dependent variable, and a spatial error component.

    >>> formula = "log(CMEDV) ~ {RM**2} + AGE + <log(CMEDV) + INDUS> + &"
    
    Fit model and output the formula as well.

    >>> model, parsed_formula = spreg.from_formula(formula, boston_df, w=weights, debug=True)
    >>> print(type(model))
    <class 'spreg.error_sp.GM_Combo'>
    >>> print(parsed_formula)
    log(CMEDV) ~ {RM**2} + AGE + INDUS + {w.sparse @ `INDUS`}
    >>> np.around(model.betas, decimals=4)
    array([[ 2.4324e+00],
        [-1.2000e-03],
        [-0.0000e+00],
        [ 2.4000e-02],
        [-1.4000e-03],
        [ 1.0000e-04],
        [-8.1000e-03]])
    >>> np.around(model.z_stat, decimals=6)
    array([[ 2.0607149e+01,  0.0000000e+00],
        [-1.7615890e+00,  7.8139000e-02],
        [-1.2940000e-03,  9.9896700e-01],
        [ 1.6212263e+01,  0.0000000e+00],
        [-3.4082950e+00,  6.5400000e-04],
        [ 8.4226000e-02,  9.3287700e-01]])
    """

    # Error checking. Minimum formula size is 5, e.g. "a ~ b"
    if type(formula) != str or len(formula) < 5:
        raise ValueError("Malformed formula string")
    
    lag_start_idx = formula.find("<")
    lag_end_idx = -1
    parsed_formula = formula[:lag_start_idx] if lag_start_idx >= 0 else ""

    y_name = formula.split("~")[0].strip()  # get name of y field, including transformations
    lag_model = False  # flag set to true if spatial lag of y component found
    spatial_model = False  # flag set to true if any spatial component found

    # Parse spatial lags in formula
    while lag_start_idx > 0:
        spatial_model = True
        lag_end_idx = formula.find(">", lag_start_idx + 1)
        if lag_end_idx < 0:
            raise ValueError("Mismatched angle brackets")
        lag_str = formula[lag_start_idx + 1:lag_end_idx]  # get everything btwn brackets

        # Parse lag_str
        fields = lag_str.split(" + ")
        if y_name in fields:
            lag_model = True
            fields.remove(y_name)

        # Default includes covariates and lags
        if len(fields) > 0:
            parsed_formula += " + ".join(fields) + " + " 
            parsed_formula += expand_lag("w", fields)  # spatial weights matrix is w in this scope
        else:  # chomp hanging plus
            parsed_formula = parsed_formula[:-3]

        # Progress parser
        lag_start_idx = formula.find("<", lag_start_idx + 1)
        if lag_start_idx >= 0:
            parsed_formula += formula[lag_end_idx + 1:lag_start_idx]

    parsed_formula += formula[lag_end_idx + 1:]  # add rest of the formula

    # Parse spatial errors in formula
    lhs, rhs = parsed_formula.split(" ~ ")
    err_model = True if "&" in rhs.split(" + ") else False
    spatial_model |= err_model
    if err_model:  # remove & from formula
        no_err = noninplace_remove(rhs.split(" + "), "&")
        parsed_formula = lhs + " ~ " + (" + ".join(no_err) if type(no_err) == list else no_err)

    if spatial_model and w is None:
        raise ValueError("Requested spatial model but did not provide weights matrix")

    # Remove intercept term from formulaic (spreg adds this already)
    parsed_formula += " - 1"
    
    # Assemble model matrices
    if type(df) == gpd.GeoDataFrame:
        df = pd.DataFrame(df)
    y, X = model_matrix(parsed_formula, df)

    # Get names of parsed/transformed variables and remove
    # names from kwargs if provided
    kwargs["name_x"] = list(X.columns)
    kwargs["name_y"] = y.columns[0]

    y = np.array(y)
    X = np.array(X)  

    method = method.lower()
    if method not in ["gm", "full", "lu", "ord"]:
        raise ValueError(f"Method must be 'gm', 'full', 'lu', 'ord'; was {method}")

    if not (err_model or lag_model):
        model = OLS(y, X, w=w, **kwargs)
    elif err_model and lag_model:
        if method != "gm":
            print("Can't use ML estimation for combo model, switching to GMM")

        if skedastic == "het":
            model = GM_Combo_Het(y, X, w=w, **kwargs)
        elif skedastic == "hom":
            model = GM_Combo_Hom(y, X, w=w, **kwargs)
        else:
            model = GM_Combo(y, X, w=w, **kwargs)
    elif lag_model:
        if method == "gm":
            model = GM_Lag(y, X, w=w, **kwargs)
        else:
            model = ML_Lag(y, X, w=w, method=method, **kwargs)
    elif err_model:
        if method == "gm":
            if skedastic == "het":
                if "yend" in kwargs:
                    model = GM_Endog_Error_Het(y, X, kwargs["yend"], kwargs["q"],
                                               w=w, **kwargs)
                else:
                    model = GM_Error_Het(y, X, w=w, **kwargs)
            elif skedastic == "hom":
                if "yend" in kwargs:
                    model = GM_Endog_Error_Hom(y, X, kwargs["yend"], kwargs["q"],
                                               w=w, **kwargs)
                else:
                    model = GM_Error_Hom(y, X, w=w, **kwargs)
            else:
                if "yend" in kwargs:
                    model = GM_Endog_Error(y, X, kwargs["yend"], kwargs["q"],
                                           w=w, **kwargs)
                else:
                    model = GM_Error(y, X, w=w, **kwargs)
        else:
            model = ML_Error(y, X, w=w, method=method, **kwargs)

    if debug:
        return model, parsed_formula
    return model


# Testing
if __name__ == "__main__":
    # Imports required for example
    import spreg
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from libpysal.examples import load_example
    from libpysal.weights import Kernel, fill_diagonal

    # Load boston example
    boston = load_example("Bostonhsg")
    boston_df = gpd.read_file(boston.get_path("boston.shp"))

    # Manually transform data and set up model matrices
    boston_df["NOXSQ"] = (10 * boston_df["NOX"])**2
    boston_df["RMSQ"] = boston_df["RM"]**2
    boston_df["LOGDIS"] = np.log(boston_df["DIS"].values)
    boston_df["LOGRAD"] = np.log(boston_df["RAD"].values)
    boston_df["TRANSB"] = boston_df["B"].values / 1000
    boston_df["LOGSTAT"] = np.log(boston_df["LSTAT"].values)

    fields = ["RMSQ", "AGE", "LOGDIS", "LOGRAD", "TAX", "PTRATIO",
              "TRANSB", "LOGSTAT", "CRIM", "ZN", "INDUS", "CHAS", "NOXSQ"]
    X = boston_df[fields].values
    y = np.log(boston_df["CMEDV"].values)  # predict log corrected median house prices from covars

    # Make weights matrix and set diagonal to 0 (necessary for lag model)
    weights = Kernel(boston_df[["x", "y"]], k=50, fixed=False)
    weights = fill_diagonal(weights, 0)


    #-------------- Testing section ----------------#
    
    # Original OLS model, manually transformed fields
    formula = "log(CMEDV) ~ RMSQ + AGE + LOGDIS + LOGRAD + TAX + PTRATIO + TRANSB" + \
              " + LOGSTAT + CRIM + ZN + INDUS + CHAS + NOXSQ"
    model, parsed_formula = spreg.from_formula(formula, boston_df, debug=True,
                                               name_y="LCMEDV", name_x=fields)
    print(type(model))
    print(parsed_formula)
    print(model.summary)
    
    # OLS model, fields transformed using formulaic
    formula = "log(CMEDV) ~ {RM**2} + AGE + log(DIS) + log(RAD) + TAX + PTRATIO" + \
              " + {B/1000} + log(LSTAT) + CRIM + ZN + INDUS + CHAS + {(10*NOX)**2}"
    model, parsed_formula = spreg.from_formula(formula, boston_df, debug=True,
                                               name_y="CMEDV", name_x=fields)
    print(type(model))
    print(parsed_formula)
    print(model.summary)

    # SLX model
    # note that type(model) == spreg.OLS as SLX is just smoothing covars
    formula = "log(CMEDV) ~ {RM**2} + AGE + log(RAD) + TAX + PTRATIO" + \
              " + {B/1000} + log(LSTAT) + <CRIM + ZN + INDUS + CHAS> + {(10*NOX)**2}"
    model, parsed_formula = spreg.from_formula(formula, boston_df, w=weights, debug=True)
    print(type(model))
    print(parsed_formula)
    print(model.summary)

    # SLY model
    formula = "log(CMEDV) ~ {RM**2} + AGE + <log(CMEDV)>"
    model, parsed_formula = spreg.from_formula(formula, boston_df, w=weights, debug=True)
    print(type(model))
    print(parsed_formula)
    print(model.summary)

    # Error model
    formula = "log(CMEDV) ~ {RM**2} + AGE + TAX + PTRATIO + {B/1000}" + \
              " + log(LSTAT) + CRIM + ZN + INDUS + CHAS + {(10*NOX)**2} + &"
    model, parsed_formula = spreg.from_formula(formula, boston_df, w=weights, debug=True)
    print(type(model))
    print(parsed_formula)
    print(model.summary)

    # Error model with ML estimation
    formula = "log(CMEDV) ~ {RM**2} + AGE + TAX + PTRATIO + {B/1000}" + \
              " + log(LSTAT) + CRIM + ZN + INDUS + CHAS + {(10*NOX)**2} + &"
    model, parsed_formula = spreg.from_formula(formula, boston_df, method="full",
                                               w=weights, debug=True)
    print(type(model))
    print(parsed_formula)
    print(model.summary)


    #-------------- Regimes testing ----------------#
    # Set up some regimes
    from spopt.region import RegionKMeansHeuristic
    kmeans = RegionKMeansHeuristic(boston_df[fields].values, 5, weights)
    kmeans.solve()
    boston_df["regime"] = kmeans.labels_
