import json
import pandas as pd
from itertools import chain
from pathlib import Path
from sklearn.model_selection import ShuffleSplit
import rampwf as rw

problem_title = "{title} tabular binary classification"
Predictions = rw.prediction_types.make_multiclass(label_names=["{target_col}"])
workflow = rw.workflows.FeatureExtractorClassifierWithEDA()  # TODO Check this
# TODO properly add scores
score_types = [
    rw.score_types.Accuracy(name="acc", precision=4),
]


def get_cv(X, y, fold_idxs=None):
    train_sizes = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_splits = 100
    if fold_idxs is None:
        train_sizes_start = 0
        train_sizes_stop = len(train_sizes) + 1
    else:
        train_sizes_start = min(fold_idxs) // n_splits
        train_sizes_stop = max(fold_idxs) // n_splits + 1
    cvs = []
    for train_size in train_sizes[train_sizes_start:train_sizes_stop]:
        cv = ShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=57)
        cvs = cvs + list(cv.split(X, y))
    return [cv for cv_i, cv in enumerate(cvs) if cv_i + (train_sizes_start * n_splits) in fold_idxs]


def _read_data(path, f_name, data_label, is_train):
    if data_label is None:
        data_path = Path(path) / "data"
    else:
        data_path = Path(path) / "data" / data_label
    data = pd.read_csv(data_path / f_name)
    y_array = data["{target_col}"].to_numpy()
    if len(y_array.shape) == 1:
        y_array = y_array.reshape((len(y_array), 1))
    X_df = data.drop("{target_col}", axis=1)
    return X_df, y_array


def get_train_data(path=".", data_label=None):
    f_name = "train.csv"
    return _read_data(path, f_name, data_label, is_train=True)


def get_test_data(path=".", data_label=None):
    f_name = "test.csv"
    return _read_data(path, f_name, data_label, is_train=False)


def get_metadata(path=".", data_label=None):
    if data_label is None:
        data_path = Path(path) / "data"
    else:
        data_path = Path(path) / "data" / data_label
    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)
    return metadata


def save_submission(y_pred, data_path=".", output_path=".", suffix="test"):
    if "test" not in suffix:
        return  # we don't care about saving the training predictions
    sample_df = pd.read_csv(Path(data_path) / "data" / "sample_submission.csv")
    df = pd.DataFrame()
    df["{id_col}"] = sample_df["{id_col}"]
    df["{target_col}"] = y_pred
    output_f_name = Path(output_path) / f"submission_{{suffix}}.csv"
    print(f"Writing submissions into {{output_f_name}}")
    df.to_csv(output_f_name, index=False)
