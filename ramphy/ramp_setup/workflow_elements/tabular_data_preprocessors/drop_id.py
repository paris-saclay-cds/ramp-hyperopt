from typing import Optional, Tuple

import numpy as np
import pandas as pd
from ramphy import BaseDataPreprocessor
#from ramphy.ramp_setup.metadata import MetaData


class DataPreprocessor(BaseDataPreprocessor):
    """Drop ID column"""

    def transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray], metadata: dict
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], dict]:
        """Removes the id column"""
        X = X.drop(columns=[metadata["id_col"]])
        if metadata is not None:
            try:
                self.drop_metadata_features(metadata=metadata, features=[metadata["id_col"]])
            except:
                pass
        return X, y, metadata
