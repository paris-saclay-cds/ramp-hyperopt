from typing import Optional, Tuple

import numpy as np
import pandas as pd
from base_data_preprocessor import BaseDataPreprocessor
from ramp_setup.metadata import MetaData


class DataPreprocessor(BaseDataPreprocessor):
    def transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray], metadata: MetaData
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], MetaData]:
        """Removes the id column"""
        X = X.drop(columns=[metadata.id_name])
        if metadata is not None:
            metadata.data_description.feature_types.pop(metadata.id_name)
        return X, y, metadata
