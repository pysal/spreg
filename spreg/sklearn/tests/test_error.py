import unittest
import libpysal
import numpy as np
from spreg.sklearn import Error
from libpysal.common import RTOL


class TestErrorGM(unittest.TestCase):
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49, 1))
        X1 = []
        X1.append(db.by_col("INC"))
        X1.append(db.by_col("CRIME"))
        self.X1 = np.array(X1).T

        # Endogeneous setup
        X2 = []
        X2.append(db.by_col("INC"))
        self.X2 = np.array(X2).T

        yd = []
        yd.append(db.by_col("CRIME"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T

        self.w = libpysal.weights.Rook.from_shapefile(
            libpysal.examples.get_path("columbus.shp")
        )
        self.w.transform = "r"

    def test_gm(self):
        reg = Error(self.w)
        reg = reg.fit(self.X1, self.y)
        betas = np.array([[0.70598088], [-0.55571746]])
        np.testing.assert_allclose(reg.coef_, betas, RTOL)
        intercept = np.array([47.94371455])
        np.testing.assert_allclose(reg.intercept_, intercept, RTOL)
        indir = np.array([0.37230161])
        np.testing.assert_allclose(reg.indir_coef_, indir, RTOL)
        predy = np.array([52.9930255])
        np.testing.assert_allclose(reg.predict(self.X1)[0], predy, RTOL)
        pr2 = 0.3495097406012179
        np.testing.assert_allclose(reg.score(self.X1, self.y), pr2, RTOL)

    def test_gm_endog(self):
        reg = Error(self.w)
        reg = reg.fit(self.X2, self.y, yend=self.yd, q=self.q)
        betas = np.array([[0.46411479], [-0.66883535]])
        np.testing.assert_allclose(reg.coef_, betas, RTOL)
        intercept = np.array([55.36095292])
        np.testing.assert_allclose(reg.intercept_, intercept, RTOL)
        indir = np.array([0.38989939])
        np.testing.assert_allclose(reg.indir_coef_, indir, RTOL)
        predy = np.array([53.9074875])
        np.testing.assert_allclose(reg.predict(self.X2)[0], predy, RTOL)
        pr2 = 0.346472557570858
        np.testing.assert_allclose(reg.score(self.X2, self.y), pr2, RTOL)


class TestErrorML(unittest.TestCase):
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
        reg = Error(self.w)
        reg = reg.fit(self.X, self.y, method=method)
        betas = np.array([[4.4024], [1.7784], [-0.3781], [0.4858]])
        np.testing.assert_allclose(reg.coef_, betas, RTOL + 0.0001)
        intercept = np.array([6.1492])
        np.testing.assert_allclose(reg.intercept_, intercept, RTOL + 0.0001)
        indir = np.array([0.2991])
        np.testing.assert_allclose(reg.indir_coef_, indir, RTOL)
        predy = np.array([6.92258051])
        np.testing.assert_allclose(reg.predict(self.X)[0], predy, RTOL)
        pr2 = 0.3057664820364818
        np.testing.assert_allclose(reg.score(self.X, self.y), pr2)

    def test_dense(self):
        self._test_core(method="full")

    def test_lu(self):
        self._test_core(method="lu", tol=RTOL * 10)

    def test_ord(self):
        self._test_core(method="ord")
