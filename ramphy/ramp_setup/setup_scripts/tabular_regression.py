import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
#from ramphy.ramp_setup.metadata import load_metadata_from_json
#from ramphy.ramp_setup.metadata import make_metadata_injectable


def tabular_regression_setup(
    download_dir: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
    ramp_templates_dir: Optional[str | Path] = None,
) -> None:
    if ramp_templates_dir is None:
        ramp_templates_dir = Path(__file__).parent.parent
    problem_template_f_name = Path(ramp_templates_dir) / "problems" / "tabular_regression_problem.py"

    download_dir = Path(download_dir)
    ramp_kit_dir = Path(ramp_kit_dir)
    if ramp_data_dir is None:
        ramp_data_dir = Path(ramp_kit_dir)
    else:
        ramp_data_dir = Path(ramp_data_dir)

    ramp_kit_dir.mkdir(parents=True, exist_ok=True)

    problem_f_name = ramp_kit_dir / "problem.py"

    problem_code = open(problem_template_f_name).read()
#    metadata = load_metadata_from_json(download_dir)
    metadata = json.load(open(download_dir / "metadata.json"))
    feature_types = metadata["data_description"]["feature_types"]
    target_cols = metadata["data_description"]["target_cols"]

#    injectable_metadata = make_metadata_injectable(metadata.asdict())
#    problem_code = problem_code.format_map(injectable_metadata)
    problem_code = problem_code.format_map(metadata)
    with open(problem_f_name, "w") as f_out:
        f_out.write(problem_code)
    (ramp_data_dir / "data").mkdir(parents=True, exist_ok=True)
    (ramp_kit_dir / "submissions").mkdir(parents=True, exist_ok=True)

    train_data = pd.read_csv(download_dir / "train.csv")
    test_data = pd.read_csv(download_dir / "test.csv")
    sample_submission = pd.read_csv(download_dir / "sample_submission.csv")

    feature_values = {}
    missing_data_count = {}
    for col, col_type in feature_types.items():
        missing_data_count[col] = int(train_data[col].isna().sum() + test_data[col].isna().sum())
        if col_type == "cat" or col_type == "bin":
            # Ensure the column is treated as string to safely use .str accessor
            train_data[col] = train_data[col].astype(str)
            train_data[col] = train_data[col].str.strip()
            train_data[col] = train_data[col].replace('nan', '')
            test_data[col] = test_data[col].astype(str)
            test_data[col] = test_data[col].str.strip()
            test_data[col] = test_data[col].replace('nan', '')
            feature_values[col] = sorted(pd.concat([train_data[col], test_data[col]]).dropna().unique().tolist())
    metadata["data_description"]["feature_values"] = feature_values
    metadata["data_description"]["missing_data_count"] = missing_data_count

    # mock test labels
    # matching mean and sigma from training set
    np.random.seed(43)
    for target_col in target_cols:
        test_data[target_col] = np.random.normal(
            train_data[target_col].mean(), train_data[target_col].std())
    test_data.to_csv(ramp_data_dir / "data" / "test.csv", index=False)
    train_data.to_csv(ramp_data_dir / "data" / "train.csv", index=False)
    sample_submission.to_csv(ramp_data_dir / "data" / "sample_submission.csv", index=False)

#    metadata.save(ramp_data_dir)
    json.dump(metadata, open(ramp_data_dir / "data" / "metadata.json", "w"))

def tabular_regression_submit(
    submission: str | Path,
    regressor: str = 'xgboost',
    feature_extractor: str = 'empty',
    data_preprocessors: list[str] = ['drop_id'],
    cat_col_impute: bool = True,
    cat_col_encode: bool = True,
    num_col_impute: bool = True,
    num_col_encode: bool = True,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
    ramp_templates_dir: str | Path = "/nas/ramp-hyperopt/ramphy/ramp_setup/",
) -> None:
    ramp_templates_dir = Path(ramp_templates_dir)
    ramp_kit_dir = Path(ramp_kit_dir)
    if ramp_data_dir is None:
        ramp_data_dir = Path(ramp_kit_dir)
    else:
        ramp_data_dir = Path(ramp_data_dir)

    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))
    regressor_f_name = (
        ramp_templates_dir /
        "workflow_elements" /
        "tabular_regressors" /
        f'{regressor}.py'
    )
    regressor_code = open(regressor_f_name).read()
    regressor_code = regressor_code.format_map(metadata)
    with open(ramp_kit_dir / "submissions" / submission / "regressor.py", "w") as f_out:
        f_out.write(regressor_code)

    fe_f_name = (
        ramp_templates_dir
        / "workflow_elements"
        / "tabular_feature_extractors"
        / f'{feature_extractor}.py'
    )
    fe_code = open(fe_f_name).read()
    fe_code = fe_code.format_map(metadata)
    with open(ramp_kit_dir / "submissions" / submission / "feature_extractor.py", "w") as f_out:
        f_out.write(fe_code)

    dp_idx = 0
    for dp in data_preprocessors:
        dp_f_name = (
            ramp_templates_dir /
            "workflow_elements" /
            "tabular_data_preprocessors" /
            f"{dp}.py"
        )
        dp_code = open(dp_f_name).read()
        dp_code = dp_code.format_map(metadata)
        with open(ramp_kit_dir / "submissions" / submission / f"data_preprocessor_{dp_idx}_{dp}.py", "w") as f_out:
            f_out.write(dp_code)
        dp_idx += 1
        
    if cat_col_impute:
        dp_f_name = (
            ramp_templates_dir /
            "workflow_elements" /
            "tabular_data_preprocessors" /
            "cat_col_imputing.py"
        )
        dp_code = open(dp_f_name).read()
        for col, col_type in metadata["data_description"]["feature_types"].items():
            if col_type == "cat" and metadata["data_description"]["missing_data_count"][col] > 0:
                dp_code_formatted = dp_code.format_map(metadata | {"col": f'"{col}"'})
                with open(ramp_kit_dir / "submissions" / submission / f"data_preprocessor_{dp_idx}_{col}_cat_col_imputing.py", "w") as f_out:
                    f_out.write(dp_code_formatted)
                dp_idx += 1

    if num_col_impute:
        dp_f_name = (
            ramp_templates_dir /
            "workflow_elements" /
            "tabular_data_preprocessors" /
            "num_col_imputing.py"
        )
        dp_code = open(dp_f_name).read()
        for col, col_type in metadata["data_description"]["feature_types"].items():
            if col_type == "num" and metadata["data_description"]["missing_data_count"][col] > 0:
                dp_code_formatted = dp_code.format_map(metadata | {"col": f'"{col}"'})
                with open(ramp_kit_dir / "submissions" / submission / f"data_preprocessor_{dp_idx}_{col}_num_col_imputing.py", "w") as f_out:
                    f_out.write(dp_code_formatted)
                dp_idx += 1

    if cat_col_encode:
        dp_f_name = (
            ramp_templates_dir /
            "workflow_elements" /
            "tabular_data_preprocessors" /
            "cat_col_encoding.py"
        )
        dp_code = open(dp_f_name).read()
        for col, col_type in metadata["data_description"]["feature_types"].items():
            if col_type == "cat":
                dp_code_formatted = dp_code.format_map(metadata | {"col": f'"{col}"'})
                with open(ramp_kit_dir / "submissions" / submission / f"data_preprocessor_{dp_idx}_{col}_cat_col_encoding.py", "w") as f_out:
                    f_out.write(dp_code_formatted)
                dp_idx += 1

