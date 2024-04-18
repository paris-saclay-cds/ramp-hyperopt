from typing import Tuple

import numpy as np
import pandas as pd
import regex
from ramp_setup.metadata import MetaData


class DataPreprocessor(object):
    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: MetaData
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, MetaData]:
        X_train = X_train.drop(columns=[metadata.id_name])
        X_test = X_test.drop(columns=[metadata.id_name])
        metadata.data_description.feature_types.pop(metadata.id_name)
        return X_train, y_train, X_test, metadata
