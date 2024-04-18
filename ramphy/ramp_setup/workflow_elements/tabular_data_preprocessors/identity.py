from typing import Optional, Tuple

import numpy as np
import pandas as pd
from base_data_preprocessor import BaseDataPreprocessor
from ramphy.ramp_setup.metadata import MetaData


class DataPreprocessor(BaseDataPreprocessor):
    """Does nothing"""

    def transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray], metadata: MetaData
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], MetaData]:
        return X, y, metadata
