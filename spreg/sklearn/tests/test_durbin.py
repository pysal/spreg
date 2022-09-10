import unittest
import libpysal
import numpy as np
from spreg.sklearn import DurbinError, DurbinLag
from libpysal.common import RTOL


class TestDurbinErrorGM(unittest.TestCase):
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49, 1))
        X1 = []
        X1.append(db.by_col("INC"))
        X1.append(db.by_col("CRIME"))
        self.X1 = np.array(X1).T

        self.w = libpysal.weights.Rook.from_shapefile(
            libpysal.examples.get_path("columbus.shp")
        )
        self.w.transform = "r"

    def test_gm(self):
        reg = DurbinError(self.w)
        reg = reg.fit(self.X1, self.y)
        betas = np.array([[0.82091923, -0.5754305, 0.47808492, 0.30069387]])
        np.testing.assert_allclose(reg.coef_, betas, RTOL)
        intercept = np.array([29.46054327])
        np.testing.assert_allclose(reg.intercept_, intercept, RTOL)
        indir = np.array([0.3583284760343048])
        np.testing.assert_allclose(reg.indir_coef_, indir, RTOL)
        predy = np.array([52.76564782])
        np.testing.assert_allclose(reg.predict(self.X1)[0], predy, RTOL)
        pr2 = 0.36318094019320923
        np.testing.assert_allclose(reg.score(self.X1), pr2, RTOL)


class TestDurbinErrorML(unittest.TestCase):
    def setUp(self):
        south = libpysal.examples.load_example("South")
        db = libpysal.io.open(south.get_path("south.dbf"), "r")
        self.y_name = "HR90"
        self.y = np.array(db.by_col(self.y_name))
        self.y.shape = (len(self.y), 1)
        self.X_names = ["RD90", "PS90", "UE90", "DV90"]
        self.X = np.array([db.by_col(var) for var in self.X_names]).T
        self.w = libpysal.weights.Queen.from_shapefile(south.get_path("south.shp"))
        self.w.transform = "r"

    def _test_core(self, method="full", tol=RTOL):
        reg = DurbinError(self.w)
        reg = reg.fit(self.X, self.y, method=method)
        betas = np.array([[4.25658458, 1.39508434, -0.18120872, 0.55136866, 0.90891701, \
                           1.29462279, -0.62973558, -0.1488492]])
        np.testing.assert_allclose(reg.coef_, betas, RTOL + 0.0001)
        intercept = np.array([9.35214564])
        np.testing.assert_allclose(reg.intercept_, intercept, RTOL + 0.0001)
        indir = np.array([0.2572919547076304])
        np.testing.assert_allclose(reg.indir_coef_, indir, RTOL)
        predy = np.array([6.67177213])
        np.testing.assert_allclose(reg.predict(self.X)[0], predy, RTOL)
        pr2 = 0.3297737217121185
        np.testing.assert_allclose(reg.score(self.X), pr2)

    def test_dense(self):
        self._test_core(method="full")

    def test_lu(self):
        self._test_core(method="lu", tol=RTOL * 10)

    def test_ord(self):
        self._test_core(method="ord")


class TestDurbinLagGM(unittest.TestCase):
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
        np.testing.assert_allclose(reg.indir_coef_, indir, RTOL)
        pr2 = 0.3551928222612527
        np.testing.assert_allclose(reg.score(self.X, self.y), pr2, RTOL)


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
        np.testing.assert_allclose(reg.indir_coef_, indir, RTOL)
        predy = np.array([-0.51218398])
        np.testing.assert_allclose(reg.predict(self.X)[0], predy, RTOL)
        pr2 = 0.6133020721559487
        np.testing.assert_allclose(reg.score(self.X, self.y), pr2, RTOL)

    def test_full(self):
        self._test_core(method="full")

    def test_lu(self):
        self._test_core(method="lu")

    def test_ord(self):
        self._test_core(method="ord")
