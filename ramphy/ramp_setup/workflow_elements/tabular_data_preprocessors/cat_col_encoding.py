from typing import Tuple

import numpy as np
import pandas as pd
from ramphy import BaseDataPreprocessor
from category_encoders import BinaryEncoder
from category_encoders import CountEncoder
from category_encoders import HashingEncoder
from category_encoders import TargetEncoder
from ramphy import Hyperparameter
#from ramphy.ramp_setup.metadata import MetaData
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# RAMP START HYPERPARAMETERS
encoding_strategy = Hyperparameter(
    dtype="str", default="Binary", values=["OneHot", "Count", "Target", "Binary", "Hashing"]
)
r_features_for_hashing = Hyperparameter(dtype="float", default=0.3, values=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
# RAMP END HYPERPARAMETERS

R_FEATURES_FOR_HASHING = float(r_features_for_hashing)
ENCODING_STRATEGY = str(encoding_strategy)


class DataPreprocessor(BaseDataPreprocessor):
    """Encodes categorical feature"""

    def __init__(self):
        super().__init__()
        self.to_cache = ENCODING_STRATEGY in ["Hashing", "Count", "Target", "Binary"]

    # TODO implement transform

    def _transform(self, X, transformer, new_columns):
        X_transformed = transformer.transform(X)
        # Handling sparse matrix output
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        # Creating a DataFrame with the correct column names and index
        X_transformed_df = pd.DataFrame(X_transformed, columns=new_columns, index=X.index)
        return X_transformed_df

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        cat_cols = [col for col, col_type in metadata["data_description"]["feature_types"].items() if col_type == "cat"]
        if ENCODING_STRATEGY == "OneHot":
            transformers = [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist"), cat_cols)]
        elif ENCODING_STRATEGY == "Count":
            transformers = [("cat", CountEncoder(handle_unknown=0, min_group_size=1, cols=cat_cols), cat_cols)]
        elif ENCODING_STRATEGY == "Target":
            transformers = [("cat", TargetEncoder(handle_unknown="value", min_samples_leaf=1, cols=cat_cols), cat_cols)]
        elif ENCODING_STRATEGY == "Binary":
            transformers = [("cat", BinaryEncoder(handle_unknown="value", cols=cat_cols), cat_cols)]
        elif ENCODING_STRATEGY == "Hashing":
            n_unique_values = np.sum([len(v) for v in metadata["data_description"]["feature_values"]])
            n_features_for_hashing = max(1, int(round(R_FEATURES_FOR_HASHING * n_unique_values)))
            transformers = [
                (
                    "cat",
                    HashingEncoder(cols=cat_cols, drop_invariant=True, n_components=n_features_for_hashing),
                    cat_cols,
                )
            ]

        column_transformer = ColumnTransformer(transformers=transformers, remainder="passthrough")
        column_transformer.fit(pd.concat([X_train, X_test]))

        new_columns = column_transformer.named_transformers_["cat"].get_feature_names_out(cat_cols)
        non_cat_columns = X_train.columns.difference(cat_cols)
        new_columns = list(new_columns) + list(non_cat_columns)
        X_train = self._transform(X_train, column_transformer, new_columns)
        X_test = self._transform(X_test, column_transformer, new_columns)

        new_col_types = dict()
        for col in new_columns:
            new_col_types[col] = "cat"
        for col in non_cat_columns:
            new_col_types[col] = metadata["data_description"]["feature_types"][col]
        metadata["data_description"]["feature_types"] = new_col_types

        return X_train, y_train, X_test, metadata
