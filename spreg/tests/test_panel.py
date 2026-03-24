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
        
        expected_vm = np.array([0.180367, 1.539671])
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

        expected_betas = np.array([[ 7.652912,  4.247395,  0.938772,  0.441378, 22.563049, 46.80849 ]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-4)
        
        expected_vm = np.array([0.077874, 0.034282, 0.04252 ]) 
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-4)

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

        expected_betas = np.array([[7.587277, 4.392572, 1.058833, 0.440583, 6.020487]])
        np.testing.assert_allclose(model.betas.T, expected_betas, atol=1e-3)
        
        expected_vm = np.array([5.286577e-02, 2.920644e-02, 3.598327e-02, 2.455856e-05,
       1.347449e-03])
        np.testing.assert_allclose(model.vm.diagonal(), expected_vm, atol=1e-3)

        other = [model.logll, model.phi]
        expected_other = [-7445.639541,  0.2503416]
        np.testing.assert_allclose(other, expected_other, rtol=1e-3)

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
        expected_other = [ 1.322196e+04,  1.323927e+04, -6.607981e+03,  1.244338e+01]
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

if __name__ == "__main__":
    unittest.main()