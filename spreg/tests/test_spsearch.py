import unittest
import libpysal
import numpy as np
import spreg
from spreg import ols as OLS
from spreg  import twosls_sp as STSLSt
from spreg  import error_sp as ERROR
from libpysal.common import RTOL

class Test_stge_classic(unittest.TestCase): 
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        y = np.array([db.by_col("CRIME")])
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("HOVAL"))
        self.X = np.array(X).T
        self.w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg_tuppla = spreg.stge_classic(self.y, self.X, self.w, name_y="CRIME", name_x=["INC", "HOVAL"], name_w="rook", mprint=False)
        reg = reg_tuppla[1]
        betas = np.array([
            [45.45909249],    # CONSTANT
            [-1.04100890],    # INC 
            [-0.25953844],    # HOVAL 
            [0.41929355]      # W_CRIME
        ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([1.12058006])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([14.60539994])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([115.025451,  -3.058734,  -0.179661,  -1.784972])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)

class Test_stge_kb(unittest.TestCase): 
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        y = np.array([db.by_col("CRIME")])
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("HOVAL"))
        self.X = np.array(X).T
        self.w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg_tuppla = spreg.stge_kb(self.y, self.X, self.w, name_y="CRIME", name_x=["INC", "HOVAL"], name_w="rook", mprint=False)
        reg = reg_tuppla[1]
        betas = np.array([
             [68.61896110],
             [-1.59731083],
             [-0.27393148]
             ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([15.37943812])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([0.34654188])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 22.42482892, -0.94235135, -0.16156749 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)

class Test_stge_pre(unittest.TestCase): 
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        y = np.array([db.by_col("CRIME")])
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("HOVAL"))
        self.X = np.array(X).T
        self.w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
    
    def test_model(self):
        reg_tuppla = spreg.stge_pre(self.y, self.X, self.w, name_y="CRIME", name_x=["INC", "HOVAL"], name_w="rook", mprint=False)
        reg = reg_tuppla[1]
        betas = np.array([
            [45.45909249],
            [-1.04100890],
            [-0.25953844],
            [0.41929355]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([14.60539994])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([1.12058006])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 115.02545067, -3.05873423, -0.17966052, -1.78497201 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)

class gets_gns(unittest.TestCase): 
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        y = np.array([db.by_col("CRIME")])
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("HOVAL"))
        self.X = np.array(X).T
        self.w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
    
    def test_model(self):
        reg_tuppla = spreg.gets_gns(self.y, self.X, self.w, name_y="CRIME", name_x=["INC", "HOVAL"], name_w="rook", mprint=False)
        reg = reg_tuppla[1]
        betas = np.array([
            [68.61896110],
            [-1.59731083],
            [-0.27393148]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([15.37943812])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([0.34654188])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 22.42482892, -0.94235135, -0.16156749 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)

class gets_sdm(unittest.TestCase): 
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        y = np.array([db.by_col("CRIME")])
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("HOVAL"))
        self.X = np.array(X).T
        self.w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
    
    def test_model(self):
        reg_tuppla = spreg.gets_sdm(self.y, self.X, self.w, name_y="CRIME", name_x=["INC", "HOVAL"], name_w="rook", mprint=False)
        reg = reg_tuppla[1]
        betas = np.array([
            [68.61896110],
            [-1.59731083],
            [-0.27393148]
            ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([15.37943812])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([0.34654188])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 22.42482892, -0.94235135, -0.16156749 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)



if __name__ == '__main__':
    unittest.main()
        
