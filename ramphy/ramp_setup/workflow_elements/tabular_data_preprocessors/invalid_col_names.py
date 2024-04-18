import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from base_data_preprocessor import BaseDataPreprocessor
from ramphy.ramp_setup.metadata import MetaData


class DataPreprocessor(BaseDataPreprocessor):
    """Modifies the invalid column names"""

    def transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray], metadata: MetaData
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], MetaData]:
        return X, y, metadata

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: MetaData
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, MetaData]:
        new_cols = {col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in metadata.data_description.feature_types}
        X_train = X_train.rename(columns=new_cols)
        X_test = X_test.rename(columns=new_cols)
        metadata.data_description.feature_types = {
            new_cols[col]: col_type for col, col_type in metadata.data_description.feature_types.items()
        }
        metadata.data_description.features = list(new_cols.keys())
        if metadata.data_description.feature_values is not None:
            metadata.data_description.feature_values = {
                new_cols[col]: col_type for col, col_type in metadata.data_description.feature_values.items()
            }
        return X_train, y_train, X_test, metadata
