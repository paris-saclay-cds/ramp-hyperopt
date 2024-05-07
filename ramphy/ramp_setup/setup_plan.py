from pathlib import Path

from ramphy.actions import RampAction

RAMP_KIT_NAME = "kaggle_blueberry_v1"
RAMP_KITS = "/home/gpaolo/src/ramp-kits/"
RAMP_KIT_DIR = Path(RAMP_KITS) / RAMP_KIT_NAME

RAMPHY_ACTIONS_MODULE = "ramphy.actions"
RAMP_SETUP_MODULE = "ramphy.ramp_setup"

BASE_SUBMISSION = "lgbm"
REGRESSOR = BASE_SUBMISSION

plan = [
    RampAction(
        module=RAMP_SETUP_MODULE,
        name="tabular_regression_setup",
        kwargs={
            "download_dir": "/home/gpaolo/src/pangu-kits/playground-series-s3e14/data",
            "ramp_kit_dir": RAMP_KIT_DIR,
        },
    ),
    RampAction(
        module=RAMP_SETUP_MODULE,
        name="tabular_regression_columnwise_first_submit",
        kwargs={"ramp_kit_dir": RAMP_KIT_DIR, "submission": BASE_SUBMISSION, "regressor": REGRESSOR},
    ),
    RampAction(
        module=RAMPHY_ACTIONS_MODULE,
        name="train",
        kwargs={
            "ramp_kit_dir": RAMP_KIT_DIR,
            "submission": BASE_SUBMISSION,
            "fold_idxs": range(900, 903),
            "force_retrain": True,
        },
    ),
]


if __name__ == "__main__":
    for action in plan:
        action.execute()
