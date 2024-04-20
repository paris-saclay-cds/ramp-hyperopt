import json
import pandas as pd
import rampwf as rw
import ramphy.ramp_setup as rs
from pathlib import Path
#from ramp_setup.metadata import MetaData
#from ramp_setup.metadata import load_metadata_from_json

problem_title = "{title} tabular regression"
Predictions = rw.prediction_types.make_regression(
    label_names=["{data_description[target_cols]}"])
workflow = rw.workflows.TabularRegressor()
score_name = "{score_name}"
score_types = [rs.score_name_type_map[score_name](name=score_name, precision=4)]
get_cv = rw.cvs.GrowingFolds().get_cv

def _read_data(path, f_name, data_label, target_cols):
    if data_label is None:
        data_path = Path(path) / "data"
    else:
        data_path = Path(path) / "data" / data_label
    data = pd.read_csv(data_path / f_name)
    y_array = data[target_cols].to_numpy()
    if len(y_array.shape) == 1:
        y_array = y_array.reshape((len(y_array), 1))
    X_df = data.drop(target_cols, axis=1)
    return X_df, y_array


def get_train_data(path=".", data_label=None):
    return _read_data(
        path, "train.csv", data_label,
        target_cols={data_description[target_cols]}
    )


def get_test_data(path=".", data_label=None):
    return _read_data(
        path, "test.csv", data_label,
        target_cols={data_description[target_cols]}
    )


def get_metadata(path=".", data_label=None) -> dict:
    if data_label is None:
        data_path = Path(path) / "data"
    else:
        data_path = Path(path) / "data" / data_label
#    metadata = load_metadata_from_json(load_path=data_path, as_dict=False)
    metadata = json.load(open(data_path / "metadata.json"))
    return metadata


def save_submission(y_pred, data_path=".", output_path=".", suffix="test"):
    if "test" not in suffix:
        return  # we don't care about saving the training predictions
    sample_df = pd.read_csv(Path(data_path) / "data" / "sample_submission.csv")
    df = pd.DataFrame()
    df["{id_col}"] = sample_df["{id_col}"]
    df[{data_description[target_cols]}] = y_pred
    output_f_name = Path(output_path) / f"submission_{{suffix}}.csv"
    print(f"Writing submissions into {{output_f_name}}")
    df.to_csv(output_f_name, index=False)
