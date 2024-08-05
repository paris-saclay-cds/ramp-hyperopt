import re
import numpy as np
import pandas as pd
from typing import Tuple
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter
from sklearn.compose import ColumnTransformer
from skrub import GapEncoder


# RAMP START HYPERPARAMETERS
n_components_gap{col} = Hyperparameter(
    dtype="int", default=10, values=[1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100]
)
# RAMP END HYPERPARAMETERS
N_COMPONENTS_GAP = int(n_components_gap{col})

% broken
class DataPreprocessor(rs.BaseDataPreprocessor):
    """Encodes text feature"""

    def __init__(self, col={str_col}):
        self.to_cache = True
        self.col = col

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        X = pd.concat((X_train, X_test))
        transformer = GapEncoder(n_components=N_COMPONENTS_GAP, random_state=1)
        transformer.fit(pd.Series(X[[self.col]].stack()))
        new_columns = transformer.get_feature_names_out(n_labels=N_COMPONENTS_GAP)
        converted_columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) + f"_{{i}}" for i, col in enumerate(new_columns)]
        X_transformed = transformer.transform(pd.Series(X[[self.col]].stack()))
        # Handling sparse matrix output
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        # Creating a DataFrame with the correct column names and index
        X_transformed_df = pd.DataFrame(X_transformed, columns=converted_columns, index=X.index)
        X = pd.concat((X, X_transformed_df), axis=1)

        metadata["data_description"]["feature_types"].pop(self.col)
        for col in converted_columns:
            metadata["data_description"]["feature_types"][col] = "num"

        X = X.drop(columns=[self.col])
        X_train = X.iloc[:len(X_train)]
        X_test = X.iloc[len(X_train):]
    
        return X_train, y_train, X_test, metadata
