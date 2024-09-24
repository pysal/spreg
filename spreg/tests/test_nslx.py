import unittest
import numpy as np
import libpysal
import spreg
import geopandas as gpd
RTOL = 1e-04

class TestNSLX(unittest.TestCase):
    def setUp(self):
        csdoh = libpysal.examples.load_example('chicagoSDOH')
        dfs = gpd.read_file(csdoh.get_path('Chi-SDOH.shp'))
        self.y = dfs[['HIS_ct']]
        self.x = dfs[['Blk14P','Hisp14P','EP_NOHSDP']]
        self.coords = dfs[["COORD_X","COORD_Y"]]

    def test_nslx_slxvars(self):
        reg = spreg.NSLX(self.y, self.x, self.coords, var_flag=1,
        slx_vars=[False,False,True], params=[(6,np.inf,"exponential")])
        np.testing.assert_allclose(reg.betas, 
            np.array([17.878828,  0.180593,  0.056209,  0.647127,  6.969201]), rtol=RTOL)
        vm = np.array([[ 1.91361545e-01, -2.09518978e-03, -2.89344531e-03,  1.50324352e-04,
            0.00000000e+00],
            [-2.09518978e-03, 6.58549881e-05,  9.80509736e-05, -1.50773218e-04,
            0.00000000e+00],
            [-2.89344531e-03,  9.80509736e-05,  2.35720689e-04, -3.57313408e-04,
            0.00000000e+00],
            [ 1.50324352e-04, -1.50773218e-04, -3.57313408e-04,  7.66414008e-04,
            0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            3.41278119e-02]])
        np.testing.assert_allclose(reg.vm, vm,RTOL)  

if __name__ == '__main__':
    unittest.main()
