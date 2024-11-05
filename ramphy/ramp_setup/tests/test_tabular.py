import copy

import numpy as np
import pandas as pd

from ramphy import ramp_setup as rs


def test_label_encoding():
    n_samples = 100
    n_features = 3
    unique_labels = np.array(["y1", "y2", "y3"])
    train_data = np.concatenate(
        [
            np.random.normal(size=(n_samples, n_features)),
            np.random.choice(unique_labels, size=(n_samples, 1)),
        ],
        axis=1,
    )
    train_data = pd.DataFrame(
        data=train_data,
        columns=["a", "b", "c", "y"],
    )
    test_data = np.concatenate(
        [
            np.random.normal(size=(n_samples, n_features)),
            np.random.choice(unique_labels, size=(n_samples, 1)),
        ],
        axis=1,
    )
    test_data = pd.DataFrame(
        data=test_data,
        columns=["a", "b", "c", "y"],
    )
    # deepcopy for test as otherwise it will modify inplace
    train_data_new, test_data_new = rs.tabular.label_encoding(
        copy.deepcopy(train_data), copy.deepcopy(test_data), ['y'], "")
    # non target columns are not touched
    pd.testing.assert_frame_equal(train_data_new.iloc[:, :-1], train_data.iloc[:, :-1])
    pd.testing.assert_frame_equal(test_data_new.iloc[:, :-1], test_data.iloc[:, :-1])
    assert train_data_new.iloc[:, -1].isin(list(range(len(unique_labels)))).all()
    assert test_data_new.iloc[:, -1].isin(list(range(len(unique_labels)))).all()

    # testing positive target values
    train_data_new, test_data_new = rs.tabular.label_encoding(
        copy.deepcopy(train_data), copy.deepcopy(test_data), ["y"], {"y": "y2"}
    )
    # non target columns are not touched
    pd.testing.assert_frame_equal(train_data_new.iloc[:, :-1], train_data.iloc[:, :-1])
    pd.testing.assert_frame_equal(test_data_new.iloc[:, :-1], test_data.iloc[:, :-1])
    assert train_data_new.iloc[:, -1].isin(list(range(len(unique_labels)))).all()
    assert test_data_new.iloc[:, -1].isin(list(range(len(unique_labels)))).all()
    train_positive_mask = train_data['y'] == "y2"
    test_positive_mask = test_data['y'] == "y2"
    assert train_data_new.loc[train_positive_mask, "y"].isin([1]).all()
    assert test_data_new.loc[test_positive_mask, "y"].isin([1]).all()
    assert train_data_new.loc[~train_positive_mask, "y"].isin([0, 2]).all()
    assert test_data_new.loc[~test_positive_mask, "y"].isin([0, 2]).all()


def test_create_dummy_targets_regression():
    # without targets in the test set
    n_samples = 100
    n_features = 3
    train_data = pd.DataFrame(
        data=np.random.normal(size=(n_samples, n_features + 1)),
        columns=['a', 'b', 'c', 'y'])
    test_data = pd.DataFrame(
        data=np.random.normal(size=(n_samples, n_features)),
        columns=['a', 'b', 'c'])
    # deepcopy for test as otherwise it will modify inplace
    train_data_orig = copy.deepcopy(train_data)
    test_data_orig = copy.deepcopy(test_data)
    test_data = rs.tabular.create_dummy_targets(
        train_data, test_data, ['y'], 'regression'
    )
    assert 'y' in test_data.columns
    assert test_data.shape[1] == n_features + 1
    pd.testing.assert_frame_equal(
        test_data_orig.iloc[:, :n_features], test_data.iloc[:, :n_features])
    # training data is not changed
    pd.testing.assert_frame_equal(train_data_orig, train_data)

    # with targets in the test set
    train_data_orig = pd.DataFrame(
        data=np.random.normal(size=(n_samples, n_features + 1)),
        columns=["a", "b", "c", "y"],
    )
    test_data_orig = pd.DataFrame(
        data=np.random.normal(size=(n_samples, n_features + 1)),
        columns=["a", "b", "c", "y"]
    )
    # deepcopy for test as otherwise it will modify inplace
    train_data_orig = copy.deepcopy(train_data)
    test_data_orig = copy.deepcopy(test_data)
    test_data = rs.tabular.create_dummy_targets(
        train_data, test_data,
        ["y"], "regression"
    )
    # data is not changed
    pd.testing.assert_frame_equal(test_data_orig, test_data)
    pd.testing.assert_frame_equal(train_data_orig, train_data)


def test_create_dummy_targets_classification():
    # without targets in the test set
    n_samples = 100
    n_features = 3
    train_data = np.concatenate(
        [
            np.random.normal(size=(n_samples, n_features)),
            np.random.choice(np.array(["y1", "y2", "y3"]), size=(n_samples, 1)),
        ],
        axis=1,
    )
    train_data = pd.DataFrame(
        data=train_data,
        columns=["a", "b", "c", "y"],
    )
    test_data = np.random.normal(size=(n_samples, n_features))
    test_data = pd.DataFrame(
        data=test_data, columns=["a", "b", "c"]
    )
    # deepcopy for test as otherwise it will modify inplace
    train_data_orig = copy.deepcopy(train_data)
    test_data_orig = copy.deepcopy(test_data)
    test_data = rs.tabular.create_dummy_targets(
        train_data, test_data, ["y"], "classification"
    )
    assert "y" in test_data.columns
    assert test_data.shape[1] == n_features + 1
    assert test_data['y'].isin(train_data_orig['y'].unique()).all()
    pd.testing.assert_frame_equal(
        test_data_orig.iloc[:, :n_features], test_data.iloc[:, :n_features]
    )
    # training data is not changed
    pd.testing.assert_frame_equal(train_data_orig, train_data)

    # with targets in the test set
    train_data = np.concatenate(
        [
            np.random.normal(size=(n_samples, n_features)),
            np.random.choice(np.array(["y1", "y2", "y3"]), size=(n_samples, 1)),
        ],
        axis=1,
    )
    train_data = pd.DataFrame(
        data=train_data,
        columns=["a", "b", "c", "y"],
    )
    test_data = np.concatenate(
        [
            np.random.normal(size=(n_samples, n_features)),
            np.random.choice(np.array(["y1", "y2", "y3"]), size=(n_samples, 1)),
        ],
        axis=1,
    )
    test_data = pd.DataFrame(
        data=test_data,
        columns=["a", "b", "c", "y"],
    )
    # deepcopy for test as otherwise it will modify inplace
    train_data_orig = copy.deepcopy(train_data)
    test_data_orig = copy.deepcopy(test_data)
    test_data = rs.tabular.create_dummy_targets(
        train_data, test_data, ["y"], "classification"
    )
    # data is not changed
    pd.testing.assert_frame_equal(test_data_orig, test_data)
    pd.testing.assert_frame_equal(train_data_orig, train_data)
