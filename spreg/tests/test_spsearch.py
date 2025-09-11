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

        # Error model
        y_1 = np.array([db.by_col("HOVAL")])
        self.y_1 = np.reshape(y_1, (49,1))
        X_1 = []    
        X_1.append(db.by_col("INC"))
        X_1.append(db.by_col("CRIME"))
        self.X_1 = np.array(X_1).T

        # Lag Robust model
        db_2 = libpysal.io.open(libpysal.examples.get_path("baltim.dbf"), "r")
        self.ds_name_2 = "baltim.dbf"
        self.y_name_2 = "PRICE"
        self.y_2 = np.array(db_2.by_col(self.y_name_2)).T
        self.y_2.shape = (len(self.y_2), 1)
        self.x_names_2 = ["NROOM", "AGE", "SQFT"]
        self.x_2 = np.array([db_2.by_col(var) for var in self.x_names_2]).T
        ww = libpysal.io.open(libpysal.examples.get_path("baltim_q.gal"))
        self.w_2 = ww.read()
        self.w_name_2 = "baltim_q.gal"
        self.w_2.transform = "r"

    def test_model_lag(self):
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
    
    def test_model_lag_nr(self): 
        reg_tuppla = spreg.stge_classic(self.y, self.X, self.w,
                                         name_y="CRIME", name_x=["HOVAL","INC"], 
                                         name_w="rook", 
                                         mprint=False, 
                                         p_value=0.05)
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
    
    def test_model_ols(self):
        reg_tuppla = spreg.stge_classic(self.y, self.X, self.w,
                                         name_y="CRIME", name_x=["HOVAL","INC"], 
                                         name_w="rook", 
                                         mprint=False, 
                                         p_value=0.001)
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

    def test_model_error(self):
        reg_tuppla = spreg.stge_classic(self.y_1, self.X_1, self.w,
                                         name_y="HOVAL", name_x=["CRIME","INC"], 
                                         name_w="rook", 
                                         mprint=False, 
                                         p_value=0.05, w_lags=1)
        reg = reg_tuppla[1]
        betas = np.array([
            [48.01199961],
            [0.71185887],
            [-0.55967638],
            [0.41229076]
            ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([53.11385567])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([27.35314733])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        y = np.array([80.46700300])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 38.43622447
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 18.46606947
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 131.43591516, -3.57190044, -1.64608849, 0.00000000 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
    
    def test_model_lag_r(self):
        reg_tuppla = spreg.stge_classic(self.y_2, self.x_2, self.w_2,
                                         mprint=False, 
                                         p_value=0.05, w_lags=2)
        reg = reg_tuppla[1]
        betas = np.array([
            [-10.69491547],
            [3.46113955],
            [-0.16058626],
            [0.56952082],
            [0.73652415]
            ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([2.27760517])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([44.72239483])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 211
        np.testing.assert_allclose(reg.n,n)
        y = np.array([47.00000000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 44.30718009
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 23.60607684
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 36.59311693, -4.84824660, -0.17042433, 0.47324132, -0.29546295 ])
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

        # SDM
        db_2 = libpysal.io.open(libpysal.examples.get_path("baltim.dbf"), "r")
        self.ds_name_2 = "baltim.dbf"
        self.y_name_2 = "PRICE"
        self.y_2 = np.array(db_2.by_col(self.y_name_2)).T
        self.y_2.shape = (len(self.y_2), 1)
        self.x_names_2 = ["NROOM", "AGE", "SQFT"]
        self.x_2 = np.array([db_2.by_col(var) for var in self.x_names_2]).T
        ww = libpysal.io.open(libpysal.examples.get_path("baltim_q.gal"))
        self.w_2 = ww.read()
        self.w_name_2 = "baltim_q.gal"
        self.w_2.transform_2 = "r"

    def test_model_ols(self):
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
    
    def test_model_lag(self):
        reg_tuppla = spreg.stge_kb(self.y, self.X, self.w, name_y="CRIME", name_x=["HOVAL", "INC"], name_w="rook", w_lags=1, p_value=0.05)
        reg = reg_tuppla[1]
        betas = np.array([
            [45.94746499],
            [-1.05273965],
            [-0.25984195],
            [0.41045190]
            ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([14.62172209])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([1.10425791])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 121.77826764, -3.22166259, -0.18412234, -1.90649789 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
    
    def test_model_sdm(self):
        reg_tuppla = spreg.stge_kb(self.y_2, self.x_2, self.w_2, w_lags=2, p_value=0.05)
        reg = reg_tuppla[1]
        betas = np.array([
            [25.13669493],
            [3.31032872],
            [-0.17985910],
            [0.75807855],
            [-1.52175029],
            [-0.01405190],
            [0.15120104],
            [0.11468233]
            ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)    
        predy = np.array([0.20031412])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([46.79968588])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 211
        np.testing.assert_allclose(reg.n,n)
        y = np.array([47.00000000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 44.30718009
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 23.60607684
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 56.26481580, -6.37943813, -0.17007519, 0.44086851, -0.66315755, -0.03739726, 0.30521417, -0.10778492 ])
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

        # Error model
        y_1 = np.array([db.by_col("HOVAL")])
        self.y_1 = np.reshape(y_1, (49,1))
        X_1 = []    
        X_1.append(db.by_col("INC"))
        X_1.append(db.by_col("CRIME"))
        self.X_1 = np.array(X_1).T

        # SDM Model
        db_2 = libpysal.io.open(libpysal.examples.get_path("baltim.dbf"), "r")
        self.ds_name_2 = "baltim.dbf"
        self.y_name_2 = "PRICE"
        self.y_2 = np.array(db_2.by_col(self.y_name_2)).T
        self.y_2.shape = (len(self.y_2), 1)
        self.x_names_2 = ["NROOM", "AGE", "SQFT"]
        self.x_2 = np.array([db_2.by_col(var) for var in self.x_names_2]).T
        ww = libpysal.io.open(libpysal.examples.get_path("baltim_q.gal"))
        self.w_2 = ww.read()
        self.w_name_2 = "baltim_q.gal"
        self.w_2.transform_2 = "r"
    
    def test_model_error(self):
        reg_tuppla = spreg.stge_pre(self.y_1, self.X_1, self.w, name_w="rook", mprint=False,p_value=0.05, w_lags=1)
        reg = reg_tuppla[1]
        betas = np.array([
            [48.01199961],
            [0.71185887],
            [-0.55967638],
            [0.41229076]
            ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([53.11385567])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([27.35314733])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        y = np.array([80.46700300])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 38.43622447
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 18.46606947
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 131.43591516, -3.57190044, -1.64608849,0.00000000 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
    
    def test_model_lag(self):
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
    
    def test_model_ols(self):
        reg_tuppla = spreg.stge_pre(self.y, self.X, self.w, name_y="CRIME", name_x=["INC", "HOVAL"], name_w="rook", p_value=0.001, mprint=False)
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

    def test_model_sdm(self):
        reg_tuppla = spreg.stge_pre(self.y_2, self.x_2, self.w_2, p_value=0.05, w_lags=1)
        reg = reg_tuppla[1]
        betas = np.array([
            [25.06475397],
            [3.31841300],
            [-0.17931108],
            [0.75623186],
            [-1.52609854],
            [-0.01357442],
            [0.14953760],
            [0.11567105]
            ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([0.16933006])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([46.83066994])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 211
        np.testing.assert_allclose(reg.n,n)
        y = np.array([47.00000000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 44.30718009
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 23.60607684
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 60.97306024, -6.90848569, -0.20610440, 0.56236208, -0.37600796, -0.06885921, 0.41470988, -0.17291425 ])
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

        # Lag model
        db_2 = libpysal.io.open(libpysal.examples.get_path("baltim.dbf"), "r")
        self.ds_name_2 = "baltim.dbf"
        self.y_name_2 = "PRICE"
        self.y_2 = np.array(db_2.by_col(self.y_name_2)).T
        self.y_2.shape = (len(self.y_2), 1)
        self.x_names_2 = ["NROOM", "AGE", "SQFT"]
        self.x_2 = np.array([db_2.by_col(var) for var in self.x_names_2]).T
        ww = libpysal.io.open(libpysal.examples.get_path("baltim_q.gal"))
        self.w_2 = ww.read()
        self.w_name_2 = "baltim_q.gal"
        self.w_2.transform = "r"
    
    def test_model_lag(self):
        reg_tuppla = spreg.gets_gns(self.y_2, self.x_2, self.w_2, p_value=0.01, w_lags=2, mprint=False)
        reg = reg_tuppla[1]
        betas = np.array([
            [-10.69491547],
            [3.46113955],
            [-0.16058626],
            [0.56952082],
            [0.73652415]
            ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([2.27760517])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([44.72239483])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 211
        np.testing.assert_allclose(reg.n,n)
        y = np.array([47.00000000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 44.30718009
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 23.60607684
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 36.59311693, -4.84824660, -0.17042433, 0.47324132, -0.29546295 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
    
    def test_model_ols(self):
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
    
    # Lag model
        db_2 = libpysal.io.open(libpysal.examples.get_path("baltim.dbf"), "r")
        self.ds_name_2 = "baltim.dbf"
        self.y_name_2 = "PRICE"
        self.y_2 = np.array(db_2.by_col(self.y_name_2)).T
        self.y_2.shape = (len(self.y_2), 1)
        self.x_names_2 = ["NROOM", "AGE", "SQFT"]
        self.x_2 = np.array([db_2.by_col(var) for var in self.x_names_2]).T
        ww = libpysal.io.open(libpysal.examples.get_path("baltim_q.gal"))
        self.w_2 = ww.read()
        self.w_name_2 = "baltim_q.gal"
        self.w_2.transform = "r"
    
    def test_model_lag(self):
        reg_tuppla = spreg.gets_sdm(self.y_2, self.x_2, self.w_2,mprint=False,p_value=0.01, w_lags=2)
        reg = reg_tuppla[1]
        betas = np.array([
            [-10.69491547],
            [3.46113955],
            [-0.16058626],
            [0.56952082],
            [0.73652415]
            ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([2.27760517])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([44.72239483])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 211
        np.testing.assert_allclose(reg.n,n)
        y = np.array([47.00000000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 44.30718009
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 23.60607684
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 36.59311693, -4.84824660, -0.17042433, 0.47324132, -0.29546295 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL) 
    
    def test_model_error(self):
        reg_tuppla = spreg.gets_sdm(self.y, self.X, self.w, name_y="CRIME", name_x=["INC", "HOVAL"], name_w="rook", mprint=False,p_value=0.05, w_lags=1)
        reg = reg_tuppla[1]
        betas = np.array([
            [62.44622279],
            [-1.11340758],
            [-0.29889733],
            [0.54224917]
            ])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        predy = np.array([16.64888691])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        u = np.array([-0.92290691])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array([ 22.26118082, -1.02229924, 0.08216422, 0.00000000 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
    
    def test_model_ols(self):
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
        
