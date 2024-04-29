from typing import Tuple, Optional
import pandas as pd
import numpy as np
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter
from copy import deepcopy
    
class DataPreprocessor(rs.BaseDataPreprocessor):
    def __init__(self):
        self.dropped_column = '{dropped_feat}'

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        X_train = X_train.drop(self.dropped_column, axis=1)
        X_test = X_test.drop(self.dropped_column, axis=1)

        metadata = deepcopy(metadata)
        metadata["data_description"]["feature_types"].pop(self.dropped_column)
        return X_train, y_train, X_test, metadata
