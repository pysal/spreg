
import contextlib
from importlib.metadata import PackageNotFoundError, version

from .dgp import *
from .diagnostics import *
from .diagnostics_panel import *
from .diagnostics_sp import *
from .diagnostics_sur import *
from .diagnostics_tsls import *
from .error_sp import *
from .error_sp_het import *
from .error_sp_het_regimes import *
from .error_sp_hom import *
from .error_sp_hom_regimes import *
from .error_sp_regimes import *
from .ml_error import *
from .ml_error_regimes import *
from .ml_lag import *
from .ml_lag_regimes import *
from .nslx import *
from .ols import *
from .ols_regimes import *
from .panel_fe import *
from .panel_re import *
from .probit import *
from .regimes import *
from .skater_reg import *
#from .spsearch import *
from .sp_panels import *
from .sputils import *
from .sur import *
from .sur_error import *
from .sur_lag import *
from .sur_utils import *
from .twosls import *
from .twosls_regimes import *
from .twosls_sp import *
from .twosls_sp_regimes import *
from .user_output import *
from .utils import *

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("spreg")
