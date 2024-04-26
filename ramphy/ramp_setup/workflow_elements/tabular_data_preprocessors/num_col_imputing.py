import re
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from ramphy import Hyperparameter
import ramphy.ramp_setup as rs

# RAMP START HYPERPARAMETERS
impute_strategy_num{var_col} = Hyperparameter(
    dtype='str', default='mean', values=['mean', 'median', 'most_frequent', 'constant'])
fill_value_num{var_col} = Hyperparameter(
    dtype='float', default=-1.0, values=[-1.0, 0.0])
# RAMP END HYPERPARAMETERS

IMPUTE_STRATEGY_NUM = str(impute_strategy_num{var_col})
FILL_VALUE_NUM = float(fill_value_num{var_col})

class DataPreprocessor(rs.BaseDataPreprocessor):
    """Imputes missing values"""

    def __init__(self, col={str_col}):
        self.col = col

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        feature_types = metadata["data_description"]["feature_types"]
        imputer = SimpleImputer(strategy=IMPUTE_STRATEGY_NUM, fill_value=FILL_VALUE_NUM)
        imputer.fit(pd.concat([X_train, X_test])[[self.col]])
        X_train[[self.col]] = imputer.transform(X_train[[self.col]])
        X_test[[self.col]] = imputer.transform(X_test[[self.col]])
        return X_train, y_train, X_test, metadata
