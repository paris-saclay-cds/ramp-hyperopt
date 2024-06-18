import os
from pathlib import Path
import ramphy as rh
import rampwf as rw
from ramphy import ramp_setup as rs

KITS_ROOT = Path("/home/gpaolo/ramp-kits")
KIT_NAME = "kaggle_heat_flux_fi_v1_1_n1"

KIT_PATH = KITS_ROOT / KIT_NAME
regressor = "xgboost"


rs.tabular_regression_columnwise_first_submit(
    submission="llm_rejector",
    regressor=regressor,
    data_preprocessors=["llm_drop_column"],
    ramp_kit_dir=KIT_PATH,
)
