import numpy as np
import pandas as pd
from typing import Tuple
import ramphy.ramp_setup as rs


class DataPreprocessor(rs.BaseDataPreprocessor):
    """Drop ID column"""

    def preprocess(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        metadata: dict,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        # the date column is sometimes the id column as well, in which case we should
        # not remove it as it contains information.
        if metadata["data_description"]["feature_types"][metadata["id_col"]] != "date":
            X_train = X_train.drop(columns=[metadata["id_col"]])
            X_test = X_test.drop(columns=[metadata["id_col"]])
        return X_train, y_train, X_test, metadata
