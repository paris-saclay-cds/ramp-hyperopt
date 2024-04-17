from pathlib import Path
from tabular_regression import tabular_regression_setup
from tabular_classification import tabular_classification_setup
from typing import Optional


def kit_setup(
    ramp_setup_kit_dir: str | Path,
    challenge_type: str,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
    ramp_templates_dir: str | Path = "/nas/ramp-hyperopt/ramphy/ramp_setup/",
) -> None:
    if challenge_type == "regression":
        return tabular_regression_setup(
            ramp_data_dir=ramp_data_dir,
            ramp_kit_dir=ramp_kit_dir,
            ramp_setup_kit_dir=ramp_setup_kit_dir,
            ramp_templates_dir=ramp_templates_dir,
        )
    elif "classification" in challenge_type:
        return tabular_classification_setup(
            ramp_data_dir=ramp_data_dir,
            ramp_kit_dir=ramp_kit_dir,
            ramp_setup_kit_dir=ramp_setup_kit_dir,
            ramp_templates_dir=ramp_templates_dir,
        )
    else:
        raise NotImplementedError("For now only regression setup is implemented!")
