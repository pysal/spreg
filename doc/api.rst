.. _api_ref:

.. currentmodule:: spreg

API reference
=============

.. _models_api:

Spatial Regression Models
-------------------------

These are the standard spatial regression models supported by the `spreg` package. Each of them contains a significant amount of detail in their docstring discussing how they're used, how they're fit, and how to interpret the results. 

.. autosummary::
   :toctree: generated/
    
    spreg.OLS
    spreg.ML_Lag
    spreg.ML_Error
    spreg.GM_Lag
    spreg.GM_Error
    spreg.GM_Error_Het
    spreg.GM_Error_Hom
    spreg.GM_Combo
    spreg.GM_Combo_Het
    spreg.GM_Combo_Hom
    spreg.GM_Endog_Error
    spreg.GM_Endog_Error_Het
    spreg.GM_Endog_Error_Hom
    spreg.TSLS
    spreg.ThreeSLS

Regimes Models
---------------

Regimes models are variants of spatial regression models which allow for structural instability in parameters. That means that these models allow different coefficient values in distinct subsets of the data. 

.. autosummary::
    :toctree: generated/

    spreg.OLS_Regimes
    spreg.ML_Lag_Regimes
    spreg.ML_Error_Regimes
    spreg.GM_Lag_Regimes
    spreg.GM_Error_Regimes
    spreg.GM_Error_Het_Regimes
    spreg.GM_Error_Hom_Regimes
    spreg.GM_Combo_Regimes
    spreg.GM_Combo_Hom_Regimes
    spreg.GM_Combo_Het_Regimes
    spreg.GM_Endog_Error_Regimes
    spreg.GM_Endog_Error_Hom_Regimes
    spreg.GM_Endog_Error_Het_Regimes

Seemingly-Unrelated Regressions
--------------------------------

Seeimingly-unrelated regression models are a generalization of linear regression. These models (and their spatial generalizations) allow for correlation in the residual terms between groups that use the same model. In spatial Seeimingly-Unrelated Regressions, the error terms across groups are allowed to exhibit a structured type of correlation: spatail correlation. 

.. autosummary::
   :toctree: generated/
    
    spreg.SUR
    spreg.SURerrorGM
    spreg.SURerrorML
    spreg.SURlagIV
    spreg.ThreeSLS
