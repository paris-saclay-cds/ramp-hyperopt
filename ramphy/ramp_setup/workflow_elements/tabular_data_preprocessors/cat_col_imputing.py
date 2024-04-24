import re
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from ramphy import Hyperparameter
import ramphy.ramp_setup as rs

# RAMP START HYPERPARAMETERS
# RAMP END HYPERPARAMETERS

class DataPreprocessor(rs.BaseDataPreprocessor):
    """Imputes missing values"""

    def __init__(self, col={col}):
        self.col = col

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        feature_types = metadata["data_description"]["feature_types"]
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(pd.concat([X_train, X_test])[[self.col]])
        X_train[[self.col]] = imputer.transform(X_train[[self.col]])
        X_test[[self.col]] = imputer.transform(X_test[[self.col]])
        return X_train, y_train, X_test, metadata
