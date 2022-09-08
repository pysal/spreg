import unittest
import numpy as np
import pandas as pd
import geopandas as gpd  # this dependency isn't required
import spreg.sklearn
import formulaic
from libpysal.examples import load_example
from libpysal.weights import Kernel, fill_diagonal
from sklearn.linear_model import LinearRegression


class TestFormula(unittest.TestCase):
    def setUp(self):
        # Load boston example
        boston = load_example("Bostonhsg")
        self.boston_df = gpd.read_file(boston.get_path("boston.shp"))

        # Make weights matrix and set diagonal to 0 (necessary for lag model)
        self.weights = Kernel(self.boston_df[["x", "y"]], k=50, fixed=False)
        self.weights = fill_diagonal(self.weights, 0)

    def tearDown(self):
        pass

    def test_empty_formula(self):
        with self.assertRaises(ValueError):
            model = spreg.sklearn.from_formula("", self.boston_df)

    def test_empty_df(self):
        with self.assertRaises(formulaic.errors.FactorEvaluationError):
            model = spreg.sklearn.from_formula("log(CMEDV) ~ ZN + CHAS", pd.DataFrame())

    def test_spatial_no_w(self):
        with self.assertRaises(ValueError):
            model = spreg.sklearn.from_formula("log(CMEDV) ~ ZN + CHAS + &", self.boston_df)

    def test_bad_method(self):
        with self.assertRaises(ValueError):
            model = spreg.sklearn.from_formula("log(CMEDV) ~ CHAS + &", self.boston_df, method="a")

    def test_debug(self):
        formula = "log(CMEDV) ~ ZN + RM"
        outputs = spreg.sklearn.from_formula(formula, self.boston_df, debug=True)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0], spreg.sklearn.from_formula(formula, self.boston_df))

    def test_init_kwargs(self):
        # OLS formula
        formula = "log(CMEDV) ~ ZN + RM"
        model = spreg.sklearn.from_formula(formula, self.boston_df,
                                           init_kwargs={"fit_intercept" : False})
        self.assertEqual(model.fit_intercept, False)

        # Lag formula
        formula = "log(CMEDV) ~ ZN + RM + <log(CMEDV)>"
        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights,
                                           init_kwargs={"fit_intercept" : False})
        self.assertEqual(model.fit_intercept, False)

        # Error formula
        formula = "log(CMEDV) ~ ZN + RM + &"
        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights,
                                           init_kwargs={"fit_intercept" : False})
        self.assertEqual(model.fit_intercept, False)

    def test_fit_kwargs(self):
        # Lag formula
        formula = "log(CMEDV) ~ ZN + RM + <log(CMEDV)>"
        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights,
                                           fit_kwargs={"w_lags" : 2})
        np.testing.assert_allclose(model.coef_, array([[0.33400965, 0.00303958]]), RTOL) 

        # Error formula
        formula = "log(CMEDV) ~ ZN + RM + &"
        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights,
                                           method="ord",
                                           fit_kwargs={"epsilon" : 1e-6})
        np.testing.assert_allclose(model.coef_, array([[0.25428286, 0.00118216]]), RTOL) 
        
    def test_nonspatial_formula(self):
        formula = "log(CMEDV) ~ ZN + AGE + RM"
        model = spreg.sklearn.from_formula(formula, self.boston_df)
        self.assertEqual(type(model), LinearRegression)

    def test_nonspatial_formula_with_transforms(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2}"
        model = spreg.sklearn.from_formula(formula, self.boston_df)
        self.assertEqual(type(model), LinearRegression)

    def test_field_not_found(self):
        with self.assertRaises(KeyError):
            model = spreg.sklearn.from_formula("log(CMEDV) ~ ZN + HI + INDUS")

    def test_spatial_lag_x(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + <CRIM>"
        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), LinearRegression)

    def test_spatial_lag_y(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + <log(CMEDV)>"
        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), spreg.sklearn.Lag)

        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights, method="lu")
        self.assertEqual(type(model), spreg.sklearn.Lag)

    def test_spatial_error(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + &"
        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), spreg.sklearn.Error)

        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights, method="lu")
        self.assertEqual(type(model), spreg.sklearn.Error)

    def test_slx_sly(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + <INDUS + log(CMEDV)>"
        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), spreg.sklearn.Lag)

        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights, method="lu")
        self.assertEqual(type(model), spreg.sklearn.Lag)

    def test_slx_error(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + <INDUS> + &"
        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), spreg.sklearn.Error)

        model = spreg.sklearn.from_formula(formula, self.boston_df, w=self.weights, method="lu")
        self.assertEqual(type(model), spreg.sklearn.Error)

    def test_pathological_formulae(self):
        with self.assertRaises(formulaic.errors.FormulaSyntaxError):
            model = spreg.sklearn.from_formula("log(CMEDV) ~ ZN + & + &", self.boston_df,
                                       w=self.weights)

        with self.assertRaises(formulaic.errors.FormulaSyntaxError):
            model = spreg.sklearn.from_formula("log(CMEDV) ~ <ZN + <AGE>>", self.boston_df,
                                       w=self.weights)
