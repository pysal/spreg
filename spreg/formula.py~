"""
Description
- adds R-style lm functionality for spatial lag and spatial error models
- add <...> operator
    - enclose lagged covariates or dependent var for assembly in model matrix
    - usage: <FIELD> adds spatial lag of that field from df to model matrix
- & adds spatial error component (& is not a reserved char in formulaic)
    - accepts sums of fields and &: <FIELD + ... + FIELD> + & is the most general format
- all terms and operators MUST be space delimited
- requires the user to have constructed a weights matrix first
    - i think this makes sense, as the functionality for this is well-documented and
      external to the actual running of the model

TODO
- figure out I/O stuff in the classes
- build dispatcher to combos (and build combo models!)
    - basic gm combo class for if user inputs lag and error, beyond that it's up to you
- add functionality for multiple estimation methods (kwarg; hook up to new classes)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from formulaic import model_matrix
from .prop_ols import OLS
from .prop_lag import Lag
from .prop_err import Error


def noninplace_remove(lst, el):
    lst.remove(el)
    return lst


def expand_lag(w_name, fields):
    """
    Creates a formula string with spatial lag terms for the given fields
    """
    fields = [fields] if type(fields) == str else fields
    output = ["{" + w_name + ".sparse @ `" + field + "`}" for field in fields]
    return " + ".join(output)


def from_formula(formula, df, w=None, method="full"):
    # Find all <...> and linearly lag every covar in there
    # if "&" is found, switch to spatial error
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
        if len(fields) > 0:
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

    # Assemble model matrices
    if type(df) == gpd.GeoDataFrame:
        df = pd.DataFrame(df)
    y, X = model_matrix(parsed_formula, df)
    y = np.array(y)
    X = np.array(X)

    if not (err_model or lag_model):
        model = OLS(X, y)
    elif lag_model:
        model = Lag(X, y, w)
    elif err_model:
        model = Error(X, y, w)

    # fit goes here

    return model, parsed_formula  # temporary return for parser demos


# Testing
if __name__ == "__main__":
    import spreg
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from libpysal.examples import load_example
    from libpysal.weights import Kernel, fill_diagonal

    boston = load_example("Bostonhsg")
    boston_df = gpd.read_file(boston.get_path("boston.shp"))

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

    weights = Kernel(boston_df[["x", "y"]], k=50, fixed=False)
    weights = fill_diagonal(weights, 0)

    # Original OLS model, manually transformed fields
    formula = "CMEDV ~ RMSQ + AGE + LOGDIS + LOGRAD + TAX + PTRATIO + TRANSB" + \
              " + LOGSTAT + CRIM + ZN + INDUS + CHAS + NOXSQ"
    model, parsed_formula = spreg.from_formula(formula, boston_df)

    # OLS model, fields transformed using formulaic
    formula = "log(CMEDV) ~ {RM**2} + AGE + log(DIS) + log(RAD) + TAX + PTRATIO" + \
              " + {B/1000} + log(LSTAT) + CRIM + ZN + INDUS + CHAS + {(10*NOX)**2}"
    model, parsed_formula = spreg.from_formula(formula, boston_df)

    # SLX model
    # note that type(model) == spreg.prop_ols.OLS as SLX is just smoothed covars
    formula = "log(CMEDV) ~ {RM**2} + AGE + log(RAD) + TAX + PTRATIO" + \
              " + {B/1000} + log(LSTAT) + <CRIM + ZN + INDUS + CHAS> + {(10*NOX)**2}"
    model, parsed_formula = spreg.from_formula(formula, boston_df, w=weights)

    # SLY model
    formula = "log(CMEDV) ~ {RM**2} + AGE + <log(CMEDV)>"
    model, parsed_formula = spreg.from_formula(formula, boston_df, w=weights)

    # Error model
    formula = "log(CMEDV) ~ {RM**2} + AGE + TAX + PTRATIO + {B/1000}" + \
              " + log(LSTAT) + CRIM + ZN + INDUS + CHAS + {(10*NOX)**2} + &"
    model, parsed_formula = spreg.from_formula(formula, boston_df, w=weights)
