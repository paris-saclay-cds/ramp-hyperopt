import numpy as np
import pandas as pd
from typing import Tuple
import ramphy.ramp_setup as rs
from category_encoders import BinaryEncoder
from category_encoders import CountEncoder
from category_encoders import HashingEncoder
from category_encoders import TargetEncoder
from ramphy import Hyperparameter
#from ramphy.ramp_setup.metadata import MetaData
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# RAMP START HYPERPARAMETERS
encoding_strategy{var_col} = Hyperparameter(
    dtype="str", default="OneHot", values=["OneHot", "Count", "Target", "Binary", "Hashing"]
)
r_features_for_hashing{var_col} = Hyperparameter(dtype="float", default=0.3, values=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
# RAMP END HYPERPARAMETERS

R_FEATURES_FOR_HASHING = float(r_features_for_hashing{var_col})
ENCODING_STRATEGY = str(encoding_strategy{var_col})


class DataPreprocessor(rs.BaseDataPreprocessor):
    """Encodes categorical feature"""

    def __init__(self, col={str_col}):
        self.to_cache = ENCODING_STRATEGY in ["Hashing", "Count", "Target", "Binary"]
        self.col = col

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        if ENCODING_STRATEGY == "OneHot":
            transformer = OneHotEncoder(handle_unknown="infrequent_if_exist")
        elif ENCODING_STRATEGY == "Count":
            transformer = CountEncoder(handle_unknown=0, min_group_size=1, cols=[self.col])
        elif ENCODING_STRATEGY == "Target":
            transformer = TargetEncoder(handle_unknown="value", min_samples_leaf=1, cols=[self.col])
        elif ENCODING_STRATEGY == "Binary":
            transformer = BinaryEncoder(handle_unknown="value", cols=[self.col])
        elif ENCODING_STRATEGY == "Hashing":
            n_unique_values = np.sum([len(v) for v in metadata["data_description"]["feature_values"]])
            n_features_for_hashing = max(1, int(round(R_FEATURES_FOR_HASHING * n_unique_values)))
            transformer = HashingEncoder(drop_invariant=True, n_components=n_features_for_hashing, cols=[self.col])

        if ENCODING_STRATEGY == "Target":
            transformer.fit(X_train[[self.col]], y_train)
        else:
            transformer.fit(pd.concat([X_train, X_test])[[self.col]])
            
        new_columns = transformer.get_feature_names_out([self.col])

        X_transformed = transformer.transform(X_train[[self.col]])
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        X_train = X_train.drop(columns=[self.col])
        X_train[new_columns] = X_transformed

        X_transformed = transformer.transform(X_test[[self.col]])
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        X_test = X_test.drop(columns=[self.col])
        X_test[new_columns] = X_transformed

        metadata["data_description"]["feature_types"].pop(self.col)
        for col in new_columns:
            metadata["data_description"]["feature_types"][col] = "num"

        return X_train, y_train, X_test, metadata
