__author__ = "Tyler D. Hoffman pysal@tdhoffman.com"

import unittest
import numpy as np
import libpysal
import spreg.sklearn
from sklearn.linear_model import LinearRegression
from libpysal.common import RTOL


class TestLMTests(unittest.TestCase):
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        y = np.array(db.by_col("HOVAL"))
        y = np.reshape(y, (49, 1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        X = np.array(X).T
        self.y = y
        self.X = X
        ols = LinearRegression()
        ols = ols.fit(self.X, self.y)
        w = libpysal.io.open(libpysal.examples.get_path("columbus.gal"), "r").read()
        w.transform = "r"
        self.w = w
        self.y_pred = ols.predict(self.X)

    def test_lm_err(self):
        lms = spreg.sklearn.lm_tests(self.y, self.y_pred, self.w, self.X)
        lme = np.array([3.097094, 0.078432])
        np.testing.assert_allclose(lms["lme"], lme, RTOL)

    def test_lm_lag(self):
        lms = spreg.sklearn.lm_tests(self.y, self.y_pred, self.w, self.X)
        lml = np.array([0.981552, 0.321816])
        np.testing.assert_allclose(lms["lml"], lml, RTOL)

    def test_rlm_err(self):
        lms = spreg.sklearn.lm_tests(self.y, self.y_pred, self.w, self.X)
        rlme = np.array([3.209187, 0.073226])
        np.testing.assert_allclose(lms["rlme"], rlme, RTOL)

    def test_rlm_lag(self):
        lms = spreg.sklearn.lm_tests(self.y, self.y_pred, self.w, self.X)
        rlml = np.array([1.093645, 0.295665])
        np.testing.assert_allclose(lms["rlml"], rlml, RTOL)

    def test_lm_sarma(self):
        lms = spreg.sklearn.lm_tests(self.y, self.y_pred, self.w, self.X)
        sarma = np.array([4.190739, 0.123025])
        np.testing.assert_allclose(lms["sarma"], sarma, RTOL)


if __name__ == "__main__":
    unittest.main()
