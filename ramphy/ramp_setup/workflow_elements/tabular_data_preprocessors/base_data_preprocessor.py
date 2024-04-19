from abc import ABC
from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd
#from ramp_setup.metadata import MetaData


class BaseDataPreprocessor(ABC):
    """Feature Preprocessor (apply to all models)."""

    def __init__(self) -> None:
        super().__init__()
        self.features_set = None

    def fit(
        self,
        X: pd.DataFrame,
        metadata: dict,
        y: np.ndarray,
    ) -> None:
        """Fit preprocessing parameters on data

        Args:
            X (pd.DataFrame): _description_
            metadata (MetaData): _description_
            y (Optional[np.ndarray]): _description_

        Returns:
            _type_: _description_
        """
        return

    def transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray], metadata: Optional[dict]
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[dict]]:
        """Transforms the data

        Args:
            X (pd.DataFrame): _description_
            y (Optional[np.ndarray]): _description_
            metadata (MetaData): _description_

        Returns:
            Tuple[pd.DataFrame, Optional[np.ndarray], MetaData]: _description_
        """

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        """Call the fit on the train data, then transform train and test data

        Args:
            X (pd.DataFrame): _description_
            y (np.ndarray): _description_
            X_test (pd.DataFrame): _description_
            metadata (Dict): _description_

        Returns:
            Tuple[ pd.DataFrame, np.ndarray, pd.DataFrame, types.ModuleType]: X_train, y, X_test, eda
        """
        # This means that it has not been trained yet
        if self.features_set is None:
            self.fit(X=X_train, metadata=metadata, y=y_train)
        X_train, y_train, metadata = self.transform(X=X_train, y=y_train, metadata=metadata)
        X_test, _, _ = self.transform(X=X_test, metadata=metadata, y=None)
        return X_train, y_train, X_test, metadata

    def drop_metadata_features(self, metadata: dict, features: list) -> dict:
        """Drops the features from the metadata

        Args:
            metadata (MetaData): _description_
            features (list): _description_

        Returns:
            MetaData: _description_
        """
        for feat in features:
            metadata["data_description"]["feature_types"].pop(feat)
#            del metadata.data_description.feature_types[feat]
#            if metadata.data_description.feature_values is not None:
#                del metadata.data_description.feature_values[feat]
        return metadata
