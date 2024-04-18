from typing import Optional, Tuple

import numpy as np
import pandas as pd
from base_data_preprocessor import BaseDataPreprocessor
from ramphy import Hyperparameter
from ramphy.ramp_setup.metadata import MetaData
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    r_regression,
    f_regression,
    mutual_info_regression,
)

# RAMP START HYPERPARAMETERS
removed_features = Hyperparameter(dtype="int", default=-1, values=[-1, -2, -3, -5, -10, -15, -20, -25])
score_function = Hyperparameter(dtype="int", default=2, values=[2, 3])
# RAMP END HYPERPARAMETERS

REMOVED_FEATURES = int(removed_features)
SCORE_FUNCTION = int(score_function)


class DataPreprocessor(BaseDataPreprocessor):
    """Uses random forest to select hte best features"""

    def fit(
        self,
        X: pd.DataFrame,
        metadata: MetaData,
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
        # We keep at least 1 feature if we ask to remove too many
        K = max(X.shape[1] + REMOVED_FEATURES, 1)

        if "classification" in metadata.task_type:
            if SCORE_FUNCTION == 1:
                scoring_function = chi2
            elif SCORE_FUNCTION == 2:
                scoring_function = f_classif
            elif SCORE_FUNCTION == 3:
                scoring_function = mutual_info_classif
            else:
                raise ValueError(
                    f"Only 3 score functions for classification are available. You asked for {SCORE_FUNCTION}."
                )
        elif "regression" in metadata.task_type:
            if SCORE_FUNCTION == 1:
                scoring_function = r_regression
            elif SCORE_FUNCTION == 2:
                scoring_function = f_regression
            elif SCORE_FUNCTION == 3:
                scoring_function = mutual_info_regression
            else:
                raise ValueError(
                    f"Only 3 score functions for regression are available. You asked for {SCORE_FUNCTION}."
                )

        self.selector = SelectKBest(scoring_function, k=K)

        if y is not None
        if metadata.task_type == "classification" and y is not None:
            y = y.flatten()
        self.selector.fit(X, y)
        # Saves the features
        self.features_set = set(self.selector.get_feature_names_out())
        self.dropped_features = list(set(X.columns) - self.features_set)

        print(f"Dropped feature: {self.dropped_features}")

    def transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray], metadata: Optional[MetaData]
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[MetaData]]:
        selected_features = list(self.selector.get_feature_names_out())
        X_prepr = self.selector.transform(X)
        X_prepr = pd.DataFrame(X_prepr, columns=selected_features)
        if metadata is not None:
            self.drop_metadata_features(metadata=metadata, features=self.dropped_features)
        return X, y, metadata
