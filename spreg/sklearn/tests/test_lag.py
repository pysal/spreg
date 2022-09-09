import unittest
import numpy as np
import libpysal
from spreg.sklearn import Lag
from libpysal.common import RTOL


class TestLagGM(unittest.TestCase):
    def setUp(self):
        self.w = libpysal.weights.Rook.from_shapefile(
            libpysal.examples.get_path("columbus.shp")
        )
        self.w.transform = "r"
        self.db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        y = np.array(self.db.by_col("HOVAL"))
        self.y = np.reshape(y, (49, 1))
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T

    def test_gm(self):
        w_lags = 2
        reg = Lag(self.w)
        reg = reg.fit(self.X, self.y, w_lags=w_lags)
        betas = np.array([[6.20888617e-01], [-4.80723451e-01]])
        np.testing.assert_allclose(reg.coef_, betas, RTOL)
        intercept = np.array([4.53017056e01])
        np.testing.assert_allclose(reg.intercept_, intercept, RTOL)
        indir = np.array([2.83622122e-02])
        np.testing.assert_allclose(reg.indir_, indir, RTOL)
        pr2 = 0.3551928222612527
        np.testing.assert_allclose(reg.score(self.X), pr2, RTOL)


class TestLagML(unittest.TestCase):
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("baltim.dbf"), "r")
        self.ds_name = "baltim.dbf"
        self.y_name = "PRICE"
        self.y = np.array(db.by_col(self.y_name)).T
        self.y.shape = (len(self.y), 1)
        self.X_names = ["NROOM", "AGE", "SQFT"]
        self.X = np.array([db.by_col(var) for var in self.X_names]).T
        ww = libpysal.io.open(libpysal.examples.get_path("baltim_q.gal"))
        self.w = ww.read()
        ww.close()
        self.w_name = "baltim_q.gal"
        self.w.transform = "r"

    def _test_core(self, method="full"):
        reg = Lag(self.w)
        reg = reg.fit(self.X, self.y, method=method)
        betas = np.array([[3.48995114], [-0.20103955], [0.65462382]])
        np.testing.assert_allclose(reg.coef_, betas, RTOL)
        intercept = np.array([-6.04040164])
        np.testing.assert_allclose(reg.intercept_, intercept, RTOL)
        indir = np.array([0.62351143])
        np.testing.assert_allclose(reg.indir_, indir, RTOL)
        predy = np.array([-0.51218398])
        np.testing.assert_allclose(reg.predict(self.X)[0], predy, RTOL)
        pr2 = 0.6133020721559487
        np.testing.assert_allclose(reg.score(self.X), pr2, RTOL)

    def test_full(self):
        self._test_core(method="full")

    def test_lu(self):
        self._test_core(method="lu")

    def test_ord(self):
        self._test_core(method="ord")
