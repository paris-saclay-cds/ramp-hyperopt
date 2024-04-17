import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ramphy as rh
import rampwf as rw


def tabular_classification_setup(
    ramp_setup_kit_dir: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
    ramp_templates_dir: str | Path = "/nas/ramp-hyperopt/ramphy/ramp_setup/",
) -> None:
    problem_template_f_name = Path(ramp_templates_dir) / "problems" / "tabular_classification_problem.py"

    ramp_setup_kit_dir = Path(ramp_setup_kit_dir)
    ramp_kit_dir = Path(ramp_kit_dir)
    if ramp_data_dir is None:
        ramp_data_dir = Path(ramp_kit_dir)
    else:
        ramp_data_dir = Path(ramp_data_dir)

    ramp_kit_dir.mkdir(parents=True, exist_ok=True)

    metadata_f_name = ramp_setup_kit_dir / "metadata.json"
    problem_f_name = ramp_kit_dir / "problem.py"

    problem_code = open(problem_template_f_name).read()
    metadata = json.load(open(metadata_f_name))
    problem_code = problem_code.format(**metadata)
    with open(problem_f_name, "w") as f_out:
        f_out.write(problem_code)
    (ramp_data_dir / "data").mkdir(parents=True, exist_ok=True)
    (ramp_kit_dir / "submissions").mkdir(parents=True, exist_ok=True)

    train_data = pd.read_csv(ramp_setup_kit_dir / "train.csv")
    test_data = pd.read_csv(ramp_setup_kit_dir / "test.csv")
    sample_submission = pd.read_csv(ramp_setup_kit_dir / "sample_submission.csv")

    metadata["col_values"] = {}
    for col, col_type in metadata["col_types"].items():
        if col_type == "cat" or col_type == "bin":
            # Ensure the column is treated as string to safely use .str accessor
            train_data[col] = train_data[col].astype(str)
            train_data[col] = train_data[col].str.strip()
            test_data[col] = test_data[col].astype(str)
            test_data[col] = test_data[col].str.strip()
            metadata["col_values"][col] = sorted(
                pd.concat([train_data[col], test_data[col]]).dropna().unique().tolist()
            )

    # TODO finish this
    # mock test labels
    # matching mean and sigma from training set
    np.random.seed(43)
    # test_data[metadata["target_col"]] = np.random.normal(
    # train_data[metadata["target_col"]].mean(), train_data[metadata["target_col"]].std()
    # )
    test_data.to_csv(ramp_data_dir / "data" / "test.csv", index=False)
    train_data.to_csv(ramp_data_dir / "data" / "train.csv", index=False)
    sample_submission.to_csv(ramp_data_dir / "data" / "sample_submission.csv", index=False)
    # if metadata["score_name"] in ["mse", "rmse"]:
    # metadata["lgbm_objective"] = "mse"
    # elif metadata["score_name"] == "mae":
    # metadata["lgbm_objective"] = "mae"

    with open(ramp_data_dir / "data" / "metadata.json", "w") as f_out:
        json.dump(metadata, f_out)
