import re
import json
import glob
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from ramphy import ramp_setup as rs
from ramphy import actions as ra
from ramphy.actions import ramp_action, RAMP_ACTIONS


@ramp_action
def tabular_regression_setup(
    download_dir: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:
    """Sets up the kit from the metadata.json, train.csv and test.csv

    Args:
        download_dir (str | Path): Dir where metadata and the data are
        ramp_kit_dir (str | Path, optional): Dir where to put the kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Dir where the kit data will be stored. Defaults to None.
    """

    problem_code = rs.utils.load_template(package=rs,
                                 template_path="problems/tabular_regression_problem.py")
    download_dir = Path(download_dir)
    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    ramp_kit_dir.mkdir(parents=True, exist_ok=True)

    problem_f_name = ramp_kit_dir / "problem.py"

    train_data = pd.read_csv(download_dir / "train.csv")
    test_data = pd.read_csv(download_dir / "test.csv")
    sample_submission = pd.read_csv(download_dir / "sample_submission.csv")

    metadata = json.load(open(download_dir / "metadata.json"))
    feature_types = metadata["data_description"]["feature_types"]
    target_cols = metadata["data_description"]["target_cols"]

    # Converting col names to a name that can become a python variable
    # and adding _<i> to col names to avoid clash
    new_feature_types = {}
    for col_i, (col, col_type) in enumerate(feature_types.items()):
        var_col = "_" + re.sub(r'[^a-zA-Z0-9_]', '_', col)
        new_col = f'{var_col}_{col_i}'
        if col == metadata["id_col"]:
            new_feature_types[col] = col_type
        else:
            new_feature_types[new_col] = col_type
    train_data = train_data.rename(
        columns=dict(zip(feature_types.keys(), new_feature_types.keys())))
    test_data = test_data.rename(
        columns=dict(zip(feature_types.keys(), new_feature_types.keys())))
    feature_types = metadata["data_description"]["feature_types"] = new_feature_types

    problem_code = problem_code.format_map(metadata)
    with open(problem_f_name, "w") as f_out:
        f_out.write(problem_code)
    (ramp_data_dir / "data").mkdir(parents=True, exist_ok=True)
    (ramp_kit_dir / "submissions").mkdir(parents=True, exist_ok=True)

    feature_values = {}
    missing_data_count = {}
    for col, col_type in feature_types.items():
        # in some challenges some columns can be in the train and not in the test and
        # vice versa (see for instance https://www.kaggle.com/c/bike-sharing-demand)
        missing_data_count[col] = 0
        if col in train_data.columns:
            missing_data_count[col] += int(train_data[col].isna().sum())
        if col in test_data.columns:
            missing_data_count[col] += int(test_data[col].isna().sum())
        if col_type == "cat" or col_type == "bin":
            # Ensure the column is treated as string to safely use .str accessor
            try:
                train_data[col] = train_data[col].astype(str)
                train_data[col] = train_data[col].str.strip()
                train_data[col] = train_data[col].replace('nan', '')
                test_data[col] = test_data[col].astype(str)
                test_data[col] = test_data[col].str.strip()
                test_data[col] = test_data[col].replace('nan', '')
                feature_values[col] = sorted(pd.concat([train_data[col], test_data[col]]).dropna().unique().tolist())
            except KeyError as e:
                print(e)
                print("Train data head:")
                print(train_data.head())
                raise

    metadata["data_description"]["feature_values"] = feature_values
    metadata["data_description"]["missing_data_count"] = missing_data_count

    # mock test labels
    # matching mean and sigma from training set
    np.random.seed(43)
    for target_col in target_cols:
        test_data[target_col] = np.random.normal(train_data[target_col].mean(), train_data[target_col].std())
    test_data.to_csv(ramp_data_dir / "data" / "test.csv", index=False)
    train_data.to_csv(ramp_data_dir / "data" / "train.csv", index=False)
    sample_submission.to_csv(ramp_data_dir / "data" / "sample_submission.csv", index=False)

    #    metadata.save(ramp_data_dir)
    json.dump(metadata, open(ramp_data_dir / "data" / "metadata.json", "w"), indent=4)


@ramp_action
def tabular_regression_submit(
    submission: str | Path,
    regressor: str = 'xgboost',
    feature_extractor: str = 'empty',
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    submission_dir = ramp_kit_dir / "submissions" / submission
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    # Regressor
    # ---------------------------
    regressor_template_path = Path("workflow_elements") / "tabular_regressors" / f'{regressor}.py'
    regressor_code = rs.utils.load_template(package=rs, template_path=regressor_template_path)
    regressor_code = regressor_code.format_map(metadata)
    with open(ramp_kit_dir / "submissions" / submission / "regressor.py", "w") as f_out:
        f_out.write(regressor_code)
    # ---------------------------

    # Feature Extractor
    # ---------------------------
    fe_template_path = Path("workflow_elements") / "tabular_feature_extractors" / f'{feature_extractor}.py'
    fe_code = rs.utils.load_template(package=rs, template_path=fe_template_path)
    fe_code = fe_code.format_map(metadata)
    with open(ramp_kit_dir / "submissions" / submission / "feature_extractor.py", "w") as f_out:
        f_out.write(fe_code)


@ramp_action
def tabular_data_preprocessors_submit(
    submission: str | Path,
    data_preprocessors: list[str] = ['drop_id'],
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:
    """Submit data preprocessor

    Args:
        submission (str | Path): New submission name
        data_preprocessors (list[str], optional): List of data preprocessors to submit. Defaults to ['drop_id'].
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    for dp in data_preprocessors:
        dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / f"{dp}.py"
        dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
        dp_code = dp_code.format_map(metadata)
        with open(ramp_kit_dir / "submissions" / submission / f"data_preprocessor_{dp_idx}_{dp}.py", "w") as f_out:
            f_out.write(dp_code)
        dp_idx += 1


@ramp_action
def tabular_cat_col_imputers_submit(
    submission: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:
    """Submit column imputer, a preprocessor that makes sure that no NaN are present in the categorical columns

    Args:
        submission (str | Path): New submission name
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "cat_col_imputing.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    for col, col_type in metadata["data_description"]["feature_types"].items():
        if col_type == "cat" and metadata["data_description"]["missing_data_count"][col] > 0:
            dp_code_formatted = dp_code.format_map(metadata | {"col": f'{col}', "str_col": f'"{col}"'})
            with open(ramp_kit_dir / "submissions" / submission / f"data_preprocessor_{dp_idx}{col}_cat_col_imputing.py", "w") as f_out:
                f_out.write(dp_code_formatted)
            dp_idx += 1


@ramp_action
def tabular_num_col_imputers_submit(
    submission: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:
    """Submit column imputer, a preprocessor that makes sure that no NaN are present in the numerical columns

    Args:
        submission (str | Path): New submission name
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "num_col_imputing.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    for col, col_type in metadata["data_description"]["feature_types"].items():
        if col_type == "num" and metadata["data_description"]["missing_data_count"][col] > 0:
            dp_code_formatted = dp_code.format_map(metadata | {"col": f'{col}', "str_col": f'"{col}"'})
            with open(ramp_kit_dir / "submissions" / submission / f"data_preprocessor_{dp_idx}{col}_num_col_imputing.py", "w") as f_out:
                f_out.write(dp_code_formatted)
            dp_idx += 1


@ramp_action
def tabular_cat_col_encoders_submit(
    submission: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:
    """Submit categorical column encoders

    Args:
        submission (str | Path): New submission name
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "cat_col_encoding.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    for col, col_type in metadata["data_description"]["feature_types"].items():
        if col_type == "cat":
            dp_code_formatted = dp_code.format_map(metadata | {"col": f'{col}', "str_col": f'"{col}"'})
            with open(ramp_kit_dir / "submissions" / submission / f"data_preprocessor_{dp_idx}{col}_cat_col_encoding.py", "w") as f_out:
                f_out.write(dp_code_formatted)
            dp_idx += 1


@ramp_action
def tabular_date_col_encoder_submit(
    submission: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:
    """Submit datetime column encoders.

    Args:
        submission (str | Path): New submission name
        ramp_kit_dir (str | Path, optional): Path of the ramp kit. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of the data dir. Defaults to None.
    """

    ramp_kit_dir, ramp_data_dir = ra.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)
    dp_template_path = (
        Path("workflow_elements") / "tabular_data_preprocessors" /
        "date_col_encoding.py"
    )
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    for col, col_type in metadata["data_description"]["feature_types"].items():
        if col_type == "date":
            dp_code_formatted = dp_code.format_map(
                metadata | {"col": f"{col}", "str_col": f'"{col}"'}
            )
            with open(
                ramp_kit_dir
                / "submissions"
                / submission
                / f"data_preprocessor_{dp_idx}_{col}_date_col_encoding.py",
                "w",
            ) as f_out:
                f_out.write(dp_code_formatted)
            dp_idx += 1


@ramp_action
def tabular_regression_columnwise_last_submit(
    submission: str | Path,
    regressor: str = 'xgboost',
    feature_extractor: str = 'empty',
    data_preprocessors: list[str] = ['drop_id'],
    cat_col_impute: bool = True,
    num_col_impute: bool = True,
    cat_col_encode: bool = True,
    num_col_encode: bool = True,
    date_col_encode: bool = True,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:
    """Make new submission with columnwise last

    Args:
        submission (str | Path): Submission name
        regressor (str, optional): Regressor. Defaults to 'xgboost'.
        feature_extractor (str, optional): FE. Defaults to 'empty'.
        data_preprocessors (list[str], optional): List of data preprocessor. Defaults to ['drop_id'].
        cat_col_impute (bool, optional): If True appends a cat_col_imputer to the list of preprocessors. Defaults to True.
        num_col_impute (bool, optional): If True appends a num_col_impute to the list of preprocessors. Defaults to True.
        cat_col_encode (bool, optional): If True appends a cat_col_encode to the list of preprocessors. Defaults to True.
        num_col_encode (bool, optional): If True appends a num_col_encode to the list of preprocessors. Defaults to True.
        date_col_encode (bool, optional): If True appends a date_col_encode to the list of preprocessors. Defaults to True.
        ramp_kit_dir (str | Path, optional): Path of kit dir. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of data dir. Defaults to None.
    """
    print(submission)
    tabular_regression_submit(
        submission=submission,
        regressor=regressor,
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
    )
    tabular_data_preprocessors_submit(
        submission=submission,
        data_preprocessors=data_preprocessors,
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
    )
    if date_col_encode:
        tabular_date_col_encoder_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    if cat_col_impute:
        tabular_cat_col_imputers_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    if num_col_impute:
        tabular_num_col_imputers_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    if cat_col_encode:
        tabular_cat_col_encoders_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    return {"created_submissions": [submission]}

@ramp_action
def tabular_regression_columnwise_first_submit(
    submission: str | Path,
    regressor: str = 'xgboost',
    feature_extractor: str = 'empty',
    data_preprocessors: list[str] = ['drop_id'],
    cat_col_impute: bool = True,
    num_col_impute: bool = True,
    cat_col_encode: bool = True,
    num_col_encode: bool = True,
    date_col_encode: bool = True,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
) -> None:
    """Make new submission with columnwise first

    Args:
        submission (str | Path): Submission name
        regressor (str, optional): Regressor. Defaults to 'xgboost'.
        feature_extractor (str, optional): FE. Defaults to 'empty'.
        data_preprocessors (list[str], optional): List of data preprocessor. Defaults to ['drop_id'].
        cat_col_impute (bool, optional): If True appends a cat_col_imputer to the list of preprocessors. Defaults to True.
        num_col_impute (bool, optional): If True appends a num_col_impute to the list of preprocessors. Defaults to True.
        cat_col_encode (bool, optional): If True appends a cat_col_encode to the list of preprocessors. Defaults to True.
        num_col_encode (bool, optional): If True appends a num_col_encode to the list of preprocessors. Defaults to True.
        date_col_encode (bool, optional): If True appends a date_col_encode to the list of preprocessors. Defaults to True.
        ramp_kit_dir (str | Path, optional): Path of kit dir. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of data dir. Defaults to None.
    """
    tabular_regression_submit(
        submission=submission,
        regressor=regressor,
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
    )
    if date_col_encode:
        tabular_date_col_encoder_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    if cat_col_impute:
        tabular_cat_col_imputers_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    if num_col_impute:
        tabular_num_col_imputers_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    if cat_col_encode:
        tabular_cat_col_encoders_submit(
            submission=submission,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
        )
    tabular_data_preprocessors_submit(
        submission=submission,
        data_preprocessors=data_preprocessors,
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
    )
    return {"created_submissions": [submission]}
