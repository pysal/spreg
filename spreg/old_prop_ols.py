class OLS(GenericModel):
    def __init__(self, X, y, robust=None, gwk=None, sig2n_k=True,
                 m=False, name_y=None, name_X=None,
                 name_gwk=None, name_ds=None):

        # Input checking
        n = USER.check_arrays(y, X)
        y = USER.check_y(y, n)
        USER.check_robust(robust, gwk)
        x_constant, name_x, warn = USER.check_constant(X, name_X)
        set_warn(self, warn)

        # Set attributes
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x_constant)
        self.robust = USER.set_robust(robust)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        self.sig2n_k = sig2n_k
        self.gwk = gwk
        super().__init__(X, y)

    def _model(self, params):
        return spdot(self.X, self.params)

    def _objective(self):
        return ((spdot(self.X, self.params) - self.y)**2).sum()

    def fit(self):
        self.xtx = spdot(self.X.T, self.X)
        xty = spdot(self.X.T, self.y)

        self.xtxi = la.inv(self.xtx)
        self.params = np.dot(self.xtxi, xty)
        return self

    def summary(self, w=None, name_w=None, nonspat_diag=True,
                spat_diag=False, moran=False, white_test=False):
        USER.check_weights(w, self.y)
        USER.check_spat_diag(spat_diag, w)
        self.name_w = USER.set_name_w(name_w, w)
        self.predy = spdot(self.X, self.params)

        self.resid = self.y - self.predy  # residuals
        self.rss = np.dot(self.resid.T, self.resid)

        if self.sig2n_k:
            self.sig2 = self.sig2n_k
        else:
            self.sig2 = self.sig2n

        if self.robust is not None:
            self.vm = ROBUST.robust_vm(reg=self, gwk=self.gwk, sig2n_k=self.sig2n_k)

        SUMMARY.OLS(reg=self, vm=self.vm, w=w, nonspat_diag=nonspat_diag,
                    spat_diag=spat_diag, moran=moran, white_test=white_test)
# if __name__ == "__main__":
    import spreg
    import numpy as np
    import geopandas as gpd
    from libpysal.examples import load_example
    boston = load_example("Bostonhsg")
    boston_df = gpd.read_file(boston.get_path("boston.shp"))

    boston_df["NOXSQ"] = (10 * boston_df["NOX"])**2
    boston_df["RMSQ"] = boston_df["RM"]**2
    boston_df["LOGDIS"] = np.log(boston_df["DIS"].values)
    boston_df["LOGRAD"] = np.log(boston_df["RAD"].values)
    boston_df["TRANSB"] = boston_df["B"].values / 1000
    boston_df["LOGSTAT"] = np.log(boston_df["LSTAT"].values)

    fields = ["RMSQ", "AGE", "LOGDIS", "LOGRAD", "TAX", "PTRATIO",
              "TRANSB", "LOGSTAT", "CRIM", "ZN", "INDUS", "CHAS", "NOXSQ"]
    X = boston_df[fields].values
    y = np.log(boston_df["CMEDV"].values)  # predict log corrected median house prices from covars

    model = spreg.OLS(X, y)
    model.fit()
    l.summary()
