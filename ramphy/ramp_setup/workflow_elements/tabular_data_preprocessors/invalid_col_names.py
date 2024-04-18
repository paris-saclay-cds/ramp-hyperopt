from typing import Tuple

import numpy as np
import pandas as pd
import re
from ramp_setup.metadata import MetaData


class DataPreprocessor(object):
    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: MetaData
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, MetaData]:
        new_cols = {{col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in metadata.data_description.feature_types}}
        X_train = X_train.rename(columns=new_cols)
        X_test = X_test.rename(columns=new_cols)
        metadata.data_description.feature_types = {
            {new_cols[col]: col_type for col, col_type in metadata.data_description.feature_types.items()}
        }
        return X_train, y_train, X_test, metadata
