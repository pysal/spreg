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

    def test_gm(self):
        w_lags = 2
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T
        # yd2, q2 = spreg.utils.set_endog(self.y, self.X, self.w, None, None, w_lags, True)
        reg = Lag(self.w)
        reg = reg.fit(self.X, self.y, w_lags=w_lags)
        betas = np.array(
            [[4.53017056e01], [6.20888617e-01], [-4.80723451e-01], [2.83622122e-02]]
        )
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        h_0 = np.array(
            [1.0, 19.531, 15.72598, 18.594, 24.7142675, 13.72216667, 27.82929567]
        )
        np.testing.assert_allclose(reg.h[0], h_0)
        hth = np.array(
            [
                49.0,
                704.371999,
                1721.312371,
                724.7435916,
                1707.35412945,
                711.31248483,
                1729.63201243,
            ]
        )
        np.testing.assert_allclose(reg.hth[0], hth, RTOL)
        hthi = np.array(
            [
                7.33701328e00,
                2.27764882e-02,
                2.18153588e-02,
                -5.11035447e-02,
                1.22515181e-03,
                -2.38079378e-01,
                -1.20149133e-01,
            ]
        )
        np.testing.assert_allclose(reg.hthi[0], hthi, RTOL)
        self.assertEqual(reg.k, 4)
        self.assertEqual(reg.kstar, 1)
        np.testing.assert_allclose(reg.mean_y, 38.436224469387746, RTOL)
        self.assertEqual(reg.n, 49)
        pfora1a2 = np.array([80.5588479, -1.06625281, -0.61703759, -1.10071931])
        np.testing.assert_allclose(reg.pfora1a2[0], pfora1a2, RTOL)
        predy_5 = np.array(
            [[50.87411532], [50.76969931], [41.77223722], [33.44262382], [28.77418036]]
        )
        np.testing.assert_allclose(reg.predy[0:5], predy_5, RTOL)
        q_5 = np.array([18.594, 24.7142675, 13.72216667, 27.82929567])
        np.testing.assert_allclose(reg.q[0], q_5)
        np.testing.assert_allclose(reg.sig2n_k, 234.54258763039289, RTOL)
        np.testing.assert_allclose(reg.sig2n, 215.39625394627919, RTOL)
        np.testing.assert_allclose(reg.sig2, 215.39625394627919, RTOL)
        np.testing.assert_allclose(reg.std_y, 18.466069465206047, RTOL)
        u_5 = np.array(
            [[29.59288768], [-6.20269831], [-15.42223722], [-0.24262282], [-5.54918036]]
        )
        np.testing.assert_allclose(reg.u[0:5], u_5, RTOL)
        np.testing.assert_allclose(reg.utu, 10554.41644336768, RTOL)
        varb = np.array(
            [
                [1.48966377e00, -2.28698061e-02, -1.20217386e-02, -1.85763498e-02],
                [-2.28698061e-02, 1.27893998e-03, 2.74600023e-04, -1.33497705e-04],
                [-1.20217386e-02, 2.74600023e-04, 1.54257766e-04, 6.86851184e-05],
                [-1.85763498e-02, -1.33497705e-04, 6.86851184e-05, 4.67711582e-04],
            ]
        )
        np.testing.assert_allclose(reg.varb, varb, RTOL)
        vm = np.array(
            [
                [3.20867996e02, -4.92607057e00, -2.58943746e00, -4.00127615e00],
                [-4.92607057e00, 2.75478880e-01, 5.91478163e-02, -2.87549056e-02],
                [-2.58943746e00, 5.91478163e-02, 3.32265449e-02, 1.47945172e-02],
                [-4.00127615e00, -2.87549056e-02, 1.47945172e-02, 1.00743323e-01],
            ]
        )
        np.testing.assert_allclose(reg.vm, vm, RTOL)
        x_0 = np.array([1.0, 19.531, 15.72598])
        np.testing.assert_allclose(reg.x[0], x_0, RTOL)
        y_5 = np.array([[80.467003], [44.567001], [26.35], [33.200001], [23.225]])
        np.testing.assert_allclose(reg.y[0:5], y_5, RTOL)
        yend_5 = np.array(
            [[35.4585005], [46.67233467], [45.36475125], [32.81675025], [30.81785714]]
        )
        np.testing.assert_allclose(reg.yend[0:5], yend_5, RTOL)
        z_0 = np.array([1.0, 19.531, 15.72598, 35.4585005])
        np.testing.assert_allclose(reg.z[0], z_0, RTOL)
        zthhthi = np.array(
            [
                [
                    1.00000000e00,
                    -2.22044605e-16,
                    -2.22044605e-16,
                    2.22044605e-16,
                    4.44089210e-16,
                    0.00000000e00,
                    -8.88178420e-16,
                ],
                [
                    0.00000000e00,
                    1.00000000e00,
                    -3.55271368e-15,
                    3.55271368e-15,
                    -7.10542736e-15,
                    7.10542736e-14,
                    0.00000000e00,
                ],
                [
                    1.81898940e-12,
                    2.84217094e-14,
                    1.00000000e00,
                    0.00000000e00,
                    -2.84217094e-14,
                    5.68434189e-14,
                    5.68434189e-14,
                ],
                [
                    -8.31133940e00,
                    -3.76104678e-01,
                    -2.07028208e-01,
                    1.32618931e00,
                    -8.04284562e-01,
                    1.30527047e00,
                    1.39136816e00,
                ],
            ]
        )
        # np.testing.assert_allclose(reg.zthhthi, zthhthi, RTOL) WHYYYY
        np.testing.assert_array_almost_equal(reg.zthhthi, zthhthi, 7)

    def test_init_white_(self):
        w_lags = 2
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T
        # yd2, q2 = spreg.utils.set_endog(self.y, self.X, self.w, None, None, w_lags, True)
        self.X = np.hstack((np.ones(self.y.shape), self.X))
        base_gm_lag = BaseGM_Lag(
            self.y, self.X, w=self.w, w_lags=w_lags, robust="white"
        )
        tbetas = np.array(
            [[4.53017056e01], [6.20888617e-01], [-4.80723451e-01], [2.83622122e-02]]
        )
        np.testing.assert_allclose(base_gm_lag.betas, tbetas)
        dbetas = D.se_betas(base_gm_lag)
        se_betas = np.array([20.47077481, 0.50613931, 0.20138425, 0.38028295])
        np.testing.assert_allclose(dbetas, se_betas)

    def test_init_hac_(self):
        w_lags = 2
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T
        # yd2, q2 = spreg.utils.set_endog(self.y, self.X, self.w, None, None, w_lags, True)
        self.X = np.hstack((np.ones(self.y.shape), self.X))
        gwk = libpysal.weights.Kernel.from_shapefile(
            libpysal.examples.get_path("columbus.shp"),
            k=15,
            function="triangular",
            fixed=False,
        )
        base_gm_lag = BaseGM_Lag(
            self.y, self.X, w=self.w, w_lags=w_lags, robust="hac", gwk=gwk
        )
        tbetas = np.array(
            [[4.53017056e01], [6.20888617e-01], [-4.80723451e-01], [2.83622122e-02]]
        )
        np.testing.assert_allclose(base_gm_lag.betas, tbetas)
        dbetas = D.se_betas(base_gm_lag)
        se_betas = np.array([19.08513569, 0.51769543, 0.18244862, 0.35460553])
        np.testing.assert_allclose(dbetas, se_betas)

    def test_init_discbd(self):
        w_lags = 2
        X = np.array(self.db.by_col("INC"))
        self.X = np.reshape(X, (49, 1))
        yd = np.array(self.db.by_col("CRIME"))
        yd = np.reshape(yd, (49, 1))
        q = np.array(self.db.by_col("DISCBD"))
        q = np.reshape(q, (49, 1))
        # yd2, q2 = spreg.utils.set_endog(self.y, self.X, self.w, yd, q, w_lags, True)
        self.X = np.hstack((np.ones(self.y.shape), self.X))
        reg = BaseGM_Lag(self.y, self.X, w=self.w, yend=yd, q=q, w_lags=w_lags)
        tbetas = np.array([[100.79359082], [-0.50215501], [-1.14881711], [-0.38235022]])
        np.testing.assert_allclose(tbetas, reg.betas)
        dbetas = D.se_betas(reg)
        se_betas = np.array([53.0829123, 1.02511494, 0.57589064, 0.59891744])
        np.testing.assert_allclose(dbetas, se_betas)

    def test_n_k(self):
        w_lags = 2
        X = []
        X.append(self.db.by_col("INC"))
        X.append(self.db.by_col("CRIME"))
        self.X = np.array(X).T
        # yd2, q2 = spreg.utils.set_endog(self.y, self.X, self.w, None, None, w_lags, True)
        self.X = np.hstack((np.ones(self.y.shape), self.X))
        reg = BaseGM_Lag(self.y, self.X, w=self.w, w_lags=w_lags, sig2n_k=True)
        betas = np.array(
            [[4.53017056e01], [6.20888617e-01], [-4.80723451e-01], [2.83622122e-02]]
        )
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        vm = np.array(
            [
                [3.49389596e02, -5.36394351e00, -2.81960968e00, -4.35694515e00],
                [-5.36394351e00, 2.99965892e-01, 6.44054000e-02, -3.13108972e-02],
                [-2.81960968e00, 6.44054000e-02, 3.61800155e-02, 1.61095854e-02],
                [-4.35694515e00, -3.13108972e-02, 1.61095854e-02, 1.09698285e-01],
            ]
        )
        np.testing.assert_allclose(reg.vm, vm, RTOL)

    def test_lag_q(self):
        w_lags = 2
        X = np.array(self.db.by_col("INC"))
        self.X = np.reshape(X, (49, 1))
        yd = np.array(self.db.by_col("CRIME"))
        yd = np.reshape(yd, (49, 1))
        q = np.array(self.db.by_col("DISCBD"))
        q = np.reshape(q, (49, 1))
        # yd2, q2 = spreg.utils.set_endog(self.y, self.X, self.w, yd, q, w_lags, False)
        self.X = np.hstack((np.ones(self.y.shape), self.X))
        reg = BaseGM_Lag(
            self.y, self.X, w=self.w, yend=yd, q=q, w_lags=w_lags, lag_q=False
        )
        tbetas = np.array([[108.83261383], [-0.48041099], [-1.18950006], [-0.56140186]])
        np.testing.assert_allclose(tbetas, reg.betas)
        dbetas = D.se_betas(reg)
        se_betas = np.array([58.33203837, 1.09100446, 0.62315167, 0.68088777])
        np.testing.assert_allclose(dbetas, se_betas)

