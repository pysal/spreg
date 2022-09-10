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
        self.w = libpysal.weights.Kernel.from_shapefile(
            libpysal.examples.get_path("boston.shp"), k=50, fixed=False
        )
        self.w.transform = "r"
        self.db = libpysal.io.open(libpysal.examples.get_path("boston.dbf"), "r")
        self.y = np.log(np.array(self.db.by_col("CMEDV")))
        X = []
        X.append(self.db.by_col("RM"))
        X.append(self.db.by_col("CRIM"))
        self.X = np.array(X).T
        self.X[:, 0] **= 2

    def test_gm(self):
        reg = DurbinLag(self.w)
        reg = reg.fit(self.X, self.y)
        betas = np.array([[-5.80084095, -0.33434679,  0.19776338, -0.01614501]])
        np.testing.assert_allclose(reg.coef_, betas, RTOL)
        intercept = np.array([188.63346406])
        np.testing.assert_allclose(reg.intercept_, intercept, RTOL)
        indir = np.array([0.03903555])
        np.testing.assert_allclose(reg.indir_coef_, indir, RTOL)
        pr2 = 0.11141781555390184
        np.testing.assert_allclose(reg.score(self.X, self.y), pr2, RTOL)


class TestLagML(unittest.TestCase):
    def setUp(self):
        self.w = libpysal.weights.Kernel.from_shapefile(
            libpysal.examples.get_path("boston.shp"), k=50, fixed=False
        )
        self.w.transform = "r"
        self.db = libpysal.io.open(libpysal.examples.get_path("boston.dbf"), "r")
        self.y = np.log(np.array(self.db.by_col("CMEDV")))
        X = []
        X.append(self.db.by_col("RM"))
        X.append(self.db.by_col("CRIM"))
        self.X = np.array(X).T
        self.X[:, 0] **= 2

    def _test_core(self, method="full"):
        reg = DurbinLag(self.w)
        reg = reg.fit(self.X, self.y, method=method)
        betas = np.array([[6.74302163, -0.17307699, -0.2677622, 0.01342728]])
        np.testing.assert_allclose(reg.coef_, betas, RTOL)
        intercept = np.array([-12.76181797])
        np.testing.assert_allclose(reg.intercept_, intercept, RTOL)
        indir = np.array([0.057416905973280974])
        np.testing.assert_allclose(reg.indir_coef_, indir, RTOL)
        pr2 = 0.0038602569028663267
        np.testing.assert_allclose(reg.score(self.X, self.y), pr2, RTOL)

    def test_full(self):
        self._test_core(method="full")

    def test_lu(self):
        self._test_core(method="lu")

    def test_ord(self):
        self._test_core(method="ord")
