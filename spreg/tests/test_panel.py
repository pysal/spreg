import unittest
import numpy as np
import libpysal
import geopandas as gpd
from spreg import panel as PANEL

class TestPanel(unittest.TestCase):
    def setUp(self):
        libpysal.examples.load_example("NCOVR")
        db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))

        east_fips = [9, 10, 11, 12, 13, 23, 24, 25, 33, 34, 36, 37, 42, 44, 45, 50, 51, 54]
        db = db[db['STFIPS'].isin(east_fips)]     

        self.y = db[['HR70','HR80','HR90']]
        self.x = db[['RD70','RD80','RD90','PS70','PS80','PS90']]
        
        self.w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
        self.w.transform = "r"

    def test_PooledOLS(self):    
        model = PANEL.PooledOLS(
            self.y, 
            self.x, 
            w=self.w,
            nonspat_diag=False,
            spat_diag=True,
            BSK_list='all',
        )

        expected_betas = np.array([[7.418732, 4.771703, 1.308517]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        
        expected_vm = np.array([0.023108, 0.018013, 0.023022])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
    
        expected_bsk = [13.526807,  19.99089, 582.610184,  11.835536,   7.77859 ]
        np.testing.assert_allclose([i for i in model.bsk['Statistic']], expected_bsk, rtol=1e-4)


    def test_PanelFE(self):
        model = PANEL.PanelFE(
            self.y, 
            self.x, 
            w=self.w, 
        )

        expected_betas = np.array([[ 0.505371, -8.026947]]) 
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        
        expected_vm = np.array([0.180595, 1.541625])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

        expected_mean_mu_i = 12.938448
        np.testing.assert_allclose(model.mean_mu_i, expected_mean_mu_i, rtol=1e-4)

    def test_PanelRE(self):
        model = PANEL.PanelRE(
            self.y, 
            self.x, 
            w=self.w, 
        )

        expected_betas = np.array([[7.57431 , 4.466996, 1.100586]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        
        expected_vm = np.array([0.037028, 0.026548, 0.036555])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

        other = [model.sigma2_mu, model.sigma2_epsilon, model.theta, model.hausman_stat]
        expected_other = [ 10.318144,  23.780203,   0.340862, 107.761451]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)

    def test_GM_ErrorPooled(self):
        model = PANEL.GM_ErrorPooled(
            self.y, 
            self.x, 
            w=self.w, 
        )

        expected_betas = np.array([[7.540989, 4.522633, 1.063156, 0.503455]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        
        expected_vm = np.array([0.07707 , 0.047983, 0.053057, 0.000786])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

    def test_ML_ErrorPooled(self):
        model = PANEL.ML_ErrorPooled(
            self.y, 
            self.x, 
            w=self.w, 
        )

        expected_betas = np.array([[7.543292, 4.516729, 1.056943, 0.453841]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        
        expected_vm = np.array([0.051193, 0.024207, 0.027356, 0.000611])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

        other = [model.aic, model.schwarz, model.logll]
        expected_other = [14839.95254 , 14857.260675, -7416.97627 ]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)

    def test_GM_ErrorRE(self):
        model = PANEL.GM_ErrorRE(
            self.y, 
            self.x, 
            w=self.w, 
        )

        expected_betas = np.array([[ 7.617791,  4.332477,  0.979866,  0.436856, 23.903141, 41.393574]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        
        expected_vm = np.array([0.067978, 0.031119, 0.037599]) 
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

    def test_GM_ErrorRE_FW(self):
        model = PANEL.GM_ErrorRE(
            self.y, 
            self.x, 
            w=self.w, 
            full_weights=True
        )

        expected_betas = np.array([[ 7.613709,  4.340652,  0.981145,  0.448915, 24.18813 , 40.364909]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)

        expected_vm = np.array([0.068669, 0.030833, 0.036947])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

    def test_GM_ErrorRE_Tau(self):
        # Tau must equal T_W in Kapoor, Kelejian and Prucha (2007)
        ws = self.w.sparse
        W = ws.toarray()
        N = W.shape[0]
        WtW = W.T @ W
        t12 = 2 * np.trace(WtW) / N
        t22 = 2 * np.trace(WtW @ WtW) / N
        t23 = np.trace(WtW @ (W.T + W)) / N
        t33 = np.trace(W @ W + WtW) / N
        expected_Tau = np.array([[2, t12, 0], [t12, t22, t23], [0, t23, t33]])
        Tau = PANEL.BaseGM_ErrorRE._get_Tau(None, ws, ws.power(2).sum())
        np.testing.assert_allclose(Tau, expected_Tau, rtol=1e-10)

    def test_ML_ErrorFE(self):
        model = PANEL.ML_ErrorFE(
            self.y, 
            self.x, 
            w=self.w, 
        )

        expected_betas = np.array([[ 0.608666, -7.681225,  0.195777]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        
        expected_vm = np.array([1.326184e-01, 1.185208e+00, 8.840893e-04])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

        other = [model.aic, model.schwarz, model.logll, model.mean_mu_i]
        expected_other = [ 1.322017e+04,  1.323171e+04, -6.608084e+03,  1.274255e+01]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)

    def test_ML_ErrorRE(self):
        model = PANEL.ML_ErrorRE(
            self.y, 
            self.x, 
            w=self.w, 
        )

        expected_betas = np.array([[7.579505, 4.408276, 1.054316, 0.47401, 4.103961]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-3)
        
        expected_vm = np.array([5.544454e-02, 2.825699e-02, 3.366076e-02, 6.821100e-04,
       4.396125e-01])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-3)

        other = [model.logll, model.phi]
        expected_other = [-7391.019693, 0.163581]
        np.testing.assert_allclose(other, expected_other, rtol=1e-3)

    def test_ML_ErrorRE_likelihood(self):
        # The concentrated likelihood must match the one computed with the
        # full covariance Omega = sigma2 [I_T kron (B'B)^-1 + ratio J_T kron I_N],
        # for a symmetric and for a nonsymmetric W.
        rng = np.random.default_rng(722)
        n, t = 16, 4
        w = libpysal.weights.lat2W(4, 4)
        w.transform = "r"
        x = np.column_stack([np.ones(n * t), rng.normal(size=(n * t, 2))])
        y = x @ np.array([[1.0], [0.8], [-0.4]]) + rng.normal(size=(n * t, 1))
        W_dir = np.random.default_rng(8).uniform(size=(n, n))
        np.fill_diagonal(W_dir, 0)
        W_dir /= W_dir.sum(axis=1, keepdims=True)
        model = PANEL.BaseML_ErrorRE.__new__(PANEL.BaseML_ErrorRE)
        model.y, model.x, model.t = y, x, t
        for W in (w.full()[0], W_dir):
            for lam, ratio in [(0.2, 0.4), (-0.3, 1.1), (0.0, 0.0)]:
                B = np.eye(n) - lam * W
                omega = np.kron(np.eye(t), np.linalg.inv(B.T @ B))
                omega += ratio * np.kron(np.ones((t, t)), np.eye(n))
                prec = np.linalg.inv(omega)
                beta = np.linalg.solve(x.T @ prec @ x, x.T @ prec @ y)
                e = y - x @ beta
                sig2 = (e.T @ prec @ e).item() / (n * t)
                expected = (n * t) / 2 * np.log(sig2) + np.linalg.slogdet(omega)[1] / 2
                actual = model.err_c_loglik_ord((lam, ratio), W)
                np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_ML_LagFE(self):
        model = PANEL.ML_LagFE(
            self.y, 
            self.x, 
            w=self.w, 
        )

        expected_betas = np.array([[ 0.523981, -7.080709,  0.193008]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        
        expected_vm = np.array([1.174713e-01, 1.018971e+00, 8.582922e-04])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

        other = [model.aic, model.schwarz, model.logll, model.mean_mu_i]
        expected_other = [ 1.322196e+04,  1.323927e+04, -6.607981e+03,  1.074278e+01]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)

        multipliers = [1.     , 0.23917, 1.23917]
        np.testing.assert_allclose(model.sp_multipliers['simple'], multipliers, atol=1e-4)

        impact_out = model.summary[-135:-85] #Check actual impact output lines
        expected_impact_out = "PS        -7.0807         -1.6935         -8.7742"
        self.assertIn(expected_impact_out, impact_out)

    def test_ML_LagRE(self):
        model = PANEL.ML_LagRE(
            self.y, 
            self.x, 
            w=self.w, 
        )

        expected_betas = np.array([[4.508465, 3.580695, 1.19552 , 0.358007, 0.742009]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-3)
        
        expected_vm = np.array([0.067993, 0.026325, 0.028039, 0.000575, 0.000455])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-3)

        other = [model.aic, model.schwarz, model.logll]
        expected_other = [14774.555911, 14797.633426, -7383.277956]
        np.testing.assert_allclose(other, expected_other, rtol=1e-3)

        multipliers = [1.      , 0.557649, 1.557649]
        np.testing.assert_allclose(model.sp_multipliers['simple'], multipliers, atol=1e-3)

        impact_out = model.summary[-135:-85] #Check actual impact output lines
        expected_impact_out = "PS         1.1955          0.6667          1.8622"
        self.assertIn(expected_impact_out, impact_out)

    def test_PooledOLS_twoway(self):
        model = PANEL.PooledOLS(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
            nonspat_diag=False,
            spat_diag=True,
            BSK_list='all',
        )
        expected_betas = np.array([[7.800349, 4.781756, 1.314681, -0.885384, -0.273779]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array([0.053031, 0.018057, 0.022965, 0.088968, 0.089289])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
        expected_bsk = [13.665972, 19.766431, 577.470591, 11.391972, 7.786688]
        np.testing.assert_allclose(
            [i for i in model.bsk['Statistic']], expected_bsk, rtol=1e-4
        )

    def test_PanelFE_twoway(self):
        model = PANEL.PanelFE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
        )
        expected_betas = np.array([[0.459332, -7.763154, -0.720306, -0.61024]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array([0.190276, 1.541609, 0.060306, 0.063254])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
        expected_mean_mu_i = 13.251921
        np.testing.assert_allclose(model.mean_mu_i, expected_mean_mu_i, rtol=1e-4)

    def test_PanelRE_twoway(self):
        model = PANEL.PanelRE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
        )
        expected_betas = np.array([[7.963289, 4.482333, 1.111048, -0.87882, -0.311604]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array([0.058266, 0.02678, 0.036517, 0.062459, 0.062936])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
        other = [model.sigma2_mu, model.sigma2_epsilon, model.theta, model.hausman_stat]
        expected_other = [10.357834, 23.661136, 0.342514, 104.883775]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)

    def test_GM_ErrorPooled_twoway(self):
        model = PANEL.GM_ErrorPooled(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
        )
        expected_betas = np.array(
            [[7.925487, 4.528931, 1.066654, -0.882939, -0.277609, 0.499603]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array([0.271308, 0.048313, 0.052779, 0.325504, 0.337523, 0.000795])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

    def test_ML_ErrorPooled_twoway(self):
        model = PANEL.ML_ErrorPooled(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
        )
        expected_betas = np.array(
            [[7.927721, 4.52292, 1.060391, -0.88297, -0.277157, 0.450746]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array([0.133564, 0.024178, 0.027279, 0.247709, 0.248142, 0.000615])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
        other = [model.aic, model.schwarz, model.logll]
        expected_other = [14840.680075, 14869.526968, -7415.340037]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)

    def test_GM_ErrorRE_twoway(self):
        model = PANEL.GM_ErrorRE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
        )
        expected_betas = np.array(
            [[8.009469, 4.338729, 0.982888, -0.879355, -0.302579, 0.434806, 23.83305, 41.442461]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array([0.131209, 0.031242, 0.037627, 0.189276, 0.189836])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

    def test_ML_ErrorFE_twoway(self):
        model = PANEL.ML_ErrorFE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
        )
        expected_betas = np.array(
            [[0.575023, -7.500393, -0.731785, -0.59552, 0.184641]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array(
            [1.368967e-01, 1.174591e+00, 5.896025e-02, 6.106609e-02, 8.940684e-04]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
        other = [model.aic, model.schwarz, model.logll, model.mean_mu_i]
        expected_other = [13214.080401, 13237.157916, -6603.040201, 13.096164]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)

    def test_ML_ErrorRE_twoway(self):
        model = PANEL.ML_ErrorRE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
        )
        expected_betas = np.array(
            [[7.967076, 4.416442, 1.058529, -0.881745, -0.289958, 0.469578, 4.130502]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-3)
        expected_vm = np.array(
            [1.304097e-01, 2.822704e-02, 3.357389e-02, 2.257601e-01, 2.262508e-01, 6.891300e-04, 4.413573e-01]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-3)
        other = [model.logll, model.phi]
        expected_other = [-7389.246958, 0.164849]
        np.testing.assert_allclose(other, expected_other, rtol=1e-3)

    def test_ML_LagFE_twoway(self):
        model = PANEL.ML_LagFE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
        )
        expected_betas = np.array(
            [[0.507792, -6.926939, -0.586646, -0.447572, 0.180828]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array(
            [1.239681e-01, 1.019751e+00, 3.981681e-02, 4.176785e-02, 8.715624e-04]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
        other = [model.aic, model.schwarz, model.logll, model.mean_mu_i]
        expected_other = [13216.599418, 13245.446311, -6603.299709, 11.117378]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)
        multipliers = [1.0, 0.220745, 1.220745]
        np.testing.assert_allclose(
            model.sp_multipliers['simple'], multipliers, atol=1e-4
        )

    def test_ML_LagRE_twoway(self):
        model = PANEL.ML_LagRE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
        )
        expected_betas = np.array(
            [[4.756492, 3.607342, 1.203576, -0.585533, -0.085171, 0.354285, 0.740364]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-3)
        expected_vm = np.array(
            [0.09327, 0.02648, 0.028083, 0.060859, 0.060958, 0.000578, 0.000453]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-3)
        other = [model.aic, model.schwarz, model.logll]
        expected_other = [14773.984357, 14814.370007, -7379.992179]
        np.testing.assert_allclose(other, expected_other, rtol=1e-3)
        multipliers = [1.0, 0.54867, 1.54867]
        np.testing.assert_allclose(
            model.sp_multipliers['simple'], multipliers, atol=1e-3
        )

    def test_PooledOLS_twoway_slx(self):
        model = PANEL.PooledOLS(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
            slx_lags=1,
            slx_vars=[True,True],
            nonspat_diag=False,
            spat_diag=False,
        )
        expected_betas = np.array(
            [[7.459689, 4.376823, 1.127946, -0.899773, -0.216858, 0.868972, 0.705451]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array(
            [0.063186, 0.039154, 0.040512, 0.088524, 0.089082, 0.062592, 0.081522]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

    def test_PanelFE_twoway_slx(self):
        model = PANEL.PanelFE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
            slx_lags=1,
        )
        expected_betas = np.array(
            [[0.674627, -7.162413, -0.697255, -0.652331, -0.725542, -1.57907]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array(
            [0.241828, 2.492844, 0.060997, 0.068755, 0.589885, 4.761883]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
        expected_mean_mu_i = 13.835659
        np.testing.assert_allclose(model.mean_mu_i, expected_mean_mu_i, rtol=1e-4)

    def test_PanelRE_twoway_slx(self):
        model = PANEL.PanelRE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
            slx_lags=1,
        )
        expected_betas = np.array(
            [[7.648183, 4.02731, 1.023444, -0.892364, -0.24771, 0.942022, 0.5487]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array(
            [0.074142, 0.053762, 0.064443, 0.062394, 0.06324 , 0.089499, 0.129028]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
        other = [model.sigma2_mu, model.sigma2_epsilon, model.theta, model.hausman_stat]
        expected_other = [ 10.206199,  23.676399,   0.339644, 103.637194]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)

    def test_GM_ErrorPooled_twoway_slx(self):
        model = PANEL.GM_ErrorPooled(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
            slx_lags=1,
        )
        expected_betas = np.array(
            [[7.444685, 4.520406, 1.002456, -0.903187, -0.224908, 0.534491, 0.825905, 0.492934]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array(
            [0.290198, 0.065433, 0.048363, 0.316966, 0.325087, 0.075384, 0.106246, 0.00081]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

    def test_ML_ErrorPooled_twoway_slx(self):
        model = PANEL.ML_ErrorPooled(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
            slx_lags=1,
        )
        expected_betas = np.array(
            [[7.44439, 4.521167, 1.000437, -0.903285, -0.225176, 0.526606, 0.827264, 0.44654]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array(
            [0.16113, 0.028954, 0.031241, 0.243432, 0.244714, 0.072921, 0.098265, 0.000619]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
        other = [model.aic, model.schwarz, model.logll]
        expected_other = [14836.904221, 14877.289871, -7411.45211]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)

    def test_GM_ErrorRE_twoway_slx(self):
        model = PANEL.GM_ErrorRE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
            slx_lags=1,
        )
        expected_betas = np.array(
            [[7.546107, 4.293166, 0.946042, -0.899351, -0.240526, 0.613014, 0.753564, 0.430928, 23.683704, 41.485409]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array(
            [0.168958, 0.037395, 0.043779, 0.18615, 0.187817, 0.095167, 0.134424]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

    def test_ML_ErrorFE_twoway_slx(self):
        model = PANEL.ML_ErrorFE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
            slx_lags=1,
        )
        expected_betas = np.array(
            [[0.674047, -7.136074, -0.708363, -0.628613, -0.568084, -1.391175, 0.183251]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array(
            [1.501966e-01, 1.519799e+00, 5.942544e-02, 6.644126e-02, 4.388733e-01, 3.387919e+00, 8.953020e-04]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
        other = [model.aic, model.schwarz, model.logll, model.mean_mu_i]
        expected_other = [13217.19551, 13251.811781, -6602.597755, 13.698298]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)

    def test_ML_ErrorRE_twoway_slx(self):
        model = PANEL.ML_ErrorRE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
            slx_lags=1,
        )
        expected_betas = np.array(
            [[7.492923, 4.317585, 0.996715, -0.902959, -0.220738, 0.710184, 0.785575, 0.463142, 4.195373]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-3)
        expected_vm = np.array(
            [1.581116e-01, 3.606320e-02, 4.119090e-02, 2.195702e-01, 2.209273e-01, 8.506672e-02, 1.188495e-01, 6.997900e-04, 4.443928e-01]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-3)
        other = [model.logll, model.phi]
        expected_other = [-7385.322259, 0.168101]
        np.testing.assert_allclose(other, expected_other, rtol=1e-3)

    def test_ML_LagFE_twoway_slx(self):
        model = PANEL.ML_LagFE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
            slx_lags=1,
        )
        expected_betas = np.array(
            [[0.713026, -7.036564, -0.575711, -0.519863, -0.698457, -0.055027, 0.183518]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        expected_vm = np.array(
            [1.572995e-01, 1.621448e+00, 4.012282e-02, 4.506673e-02, 3.837365e-01, 3.154243e+00, 8.948976e-04]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)
        other = [model.aic, model.schwarz, model.logll, model.mean_mu_i]
        expected_other = [13218.991327, 13259.376977, -6602.495663, 11.262956]
        np.testing.assert_allclose(other, expected_other, rtol=1e-4)
        multipliers = [1.0, 0.224767, 1.224767]
        np.testing.assert_allclose(
            model.sp_multipliers['simple'], multipliers, atol=1e-4
        )
        impact_out = model.summary[-270:-220]
        expected_impact_out = "PS        -7.0366         -1.6490         -8.6855\n"
        self.assertIn(expected_impact_out, impact_out)

    def test_ML_LagRE_twoway_slx(self):
        model = PANEL.ML_LagRE(
            self.y,
            self.x,
            w=self.w,
            time_effects=True,
            slx_lags=1,
        )
        expected_betas = np.array(
            [[4.467153, 4.170904, 0.934329, -0.540667, -0.126389, -1.098932, 0.156781, 0.400599, 0.760345]]
        )
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-3)
        expected_vm = np.array(
            [0.098409, 0.042185, 0.047579, 0.061178, 0.061308, 0.084525, 0.096859, 0.000672, 0.000469]
        )
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-3)
        other = [model.aic, model.schwarz, model.logll]
        expected_other = [14761.496453, 14813.420861, -7371.748227]
        np.testing.assert_allclose(other, expected_other, rtol=1e-3)
        multipliers = [1.0, 0.668331, 1.668331]
        np.testing.assert_allclose(
            model.sp_multipliers['simple'], multipliers, atol=1e-3
        )
        impact_out = model.summary[-270:-220]
        expected_impact_out = "PS         0.9343          0.8860          1.8203\n"
        self.assertIn(expected_impact_out, impact_out)
        
if __name__ == "__main__":
    unittest.main()