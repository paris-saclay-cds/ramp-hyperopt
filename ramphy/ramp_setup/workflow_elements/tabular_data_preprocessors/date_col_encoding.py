import numpy as np
import pandas as pd
from typing import Tuple
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter

# RAMP START HYPERPARAMETERS
encoding_strategy{col} = Hyperparameter(
    dtype="str", default="Raw", values=["Raw", "Cyclical", "Both"]
)
# RAMP END HYPERPARAMETERS

ENCODING_STRATEGY = str(encoding_strategy{col})


def sin_transformer(x, period):
    return np.sin(x / period * 2 * np.pi)


def cos_transformer(x, period):
    return np.cos(x / period * 2 * np.pi)


def raw_features(
    df, column_name,
    month, weekofyear, dayofyear, dayofmonth, dayofweek, hour,
    minute, second, microsecond, nanosecond):

    df[column_name + "_month"] = month
    df[column_name + "_weekofyear"] = weekofyear
    df[column_name + "_dayofyear"] = dayofyear
    df[column_name + "_dayofmonth"] = dayofmonth
    df[column_name + "_dayofweek"] = dayofweek
    df[column_name + "_hour"] = hour
    df[column_name + "_minute"] = minute
    df[column_name + "_second"] = second
    df[column_name + "_microsecond"] = microsecond
    df[column_name + "_nanosecond"] = nanosecond

    return df


def cyclical_features(
    df, column_name,
    month, weekofyear, dayofyear, dayofmonth, dayofweek, hour,
    minute, second, microsecond, nanosecond):

    df[column_name + '_month_cos'] = cos_transformer(
        month, 12
    )
    df[column_name + '_month_sin'] = sin_transformer(
        month, 12
    )
    df[column_name + '_weekofyear_cos'] = cos_transformer(
        weekofyear, 52
    )
    df[column_name + '_weekofyear_sin'] = sin_transformer(
        weekofyear, 52
    )
    df[column_name + '_dayofyear_cos'] = cos_transformer(
        dayofyear, 365.25
    )
    df[column_name + '_dayofyear_sin'] = sin_transformer(
        dayofyear, 365.25
    )
    df[column_name + '_dayofmonth_cos'] = cos_transformer(
        dayofmonth, 30  # can be 31 or 28... to be fixed
    )
    df[column_name + '_dayofmonth_sin'] = sin_transformer(
        dayofmonth, 30  # can be 31 or 28... to be fixed
    )
    df[column_name + '_dayofweek_cos'] = cos_transformer(
        dayofweek, 7
    )
    df[column_name + '_dayofweek_sin'] = sin_transformer(
        dayofweek, 7
    )
    df[column_name + '_hour_cos'] = cos_transformer(
        hour, 24
    )
    df[column_name + '_hour_sin'] = sin_transformer(
        hour, 24
    )
    df[column_name + '_minute_cos'] = cos_transformer(
        minute, 60
    )
    df[column_name + '_minute_sin'] = sin_transformer(
        minute, 60
    )
    df[column_name + '_second_cos'] = cos_transformer(
        second, 60
    )
    df[column_name + '_second_sin'] = sin_transformer(
        second, 60
    )
    df[column_name + '_microsecond_cos'] = cos_transformer(
        microsecond, 1e6
    )
    df[column_name + '_microsecond_sin'] = sin_transformer(
        microsecond, 1e6
    )
    df[column_name + '_nanosecond_cos'] = cos_transformer(
        nanosecond, 1e9
    )
    df[column_name + '_nanosecond_sin'] = sin_transformer(
        nanosecond, 1e9
    )
    return df


class DataPreprocessor(rs.BaseDataPreprocessor):
    """Encodes date column"""

    def __init__(self, col={str_col}):
        self.to_cache = ENCODING_STRATEGY in ["Raw"]
        self.col = col

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        # if we add seconds and minutes and the times are say hourly, then the second
        # and minute features will be constant and can thus be removed
        X_transformed[self.col] = pd.to_datetime(
            X_transformed[self.col]
        )
        X_transformed[self.col + "_year"] = X_transformed[
            self.col
        ].dt.year

        month = X_transformed[self.col].dt.month.to_numpy()
        weekofyear = X_transformed[self.col].dt.isocalendar().week
        dayofyear = X_transformed[self.col].dt.dayofyear.to_numpy()
        dayofmonth = X_transformed[self.col].dt.day.to_numpy()
        dayofweek = X_transformed[self.col].dt.dayofweek.to_numpy()
        hour = X_transformed[self.col].dt.hour.to_numpy()
        minute = X_transformed[self.col].dt.minute.to_numpy()
        second = X_transformed[self.col].dt.second.to_numpy()
        microsecond = X_transformed[self.col].dt.microsecond.to_numpy()
        nanosecond = X_transformed[self.col].dt.nanosecond.to_numpy()

        X_transformed['is_weekend'] = dayofweek
        X_transformed['is_weekend'] = X_transformed["is_weekend"].apply(
            lambda x: 1 if x >= 5 else 0)
        # ideally we should also add holidays but this is country/job dependent. An LLM
        # should be able to know that more or less

        if ENCODING_STRATEGY == 'Raw':
            X_transformed = raw_features(
                X_transformed, self.col,
                month, weekofyear, dayofyear, dayofmonth, dayofweek, hour,
                minute, second, microsecond, nanosecond
            )

        elif ENCODING_STRATEGY == 'Cyclical':
            X_transformed = cyclical_features(
                X_transformed, self.col,
                month, weekofyear, dayofyear, dayofmonth, dayofweek, hour,
                minute, second, microsecond, nanosecond)
        elif ENCODING_STRATEGY == 'Both':
            X_transformed = raw_features(
                X_transformed, self.col,
                month, weekofyear, dayofyear, dayofmonth, dayofweek, hour,
                minute, second, microsecond, nanosecond
            )
            X_transformed = cyclical_features(
                X_transformed, self.col,
                month, weekofyear, dayofyear, dayofmonth, dayofweek, hour,
                minute, second, microsecond, nanosecond)
        else:
            raise ValueError('Encoding strategy not supported for dates')

        # removing columns with constant value
        cols_with_unique_value = X_transformed.nunique() == 1
        X_transformed = X_transformed.loc[:, ~cols_with_unique_value]
        X_transformed.drop(self.col, axis=1, inplace=True)

        return X_transformed

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:

        X_tf = self._transform(pd.concat([X_train, X_test], axis=0)[[self.col]])
        X_train_tf = X_tf.iloc[:len(X_train)]
        X_test_tf = X_tf.iloc[len(X_train):]
        new_columns = X_tf.columns

        X_train = X_train.drop(columns=[self.col])
        X_train[new_columns] = X_train_tf
        X_test = X_test.drop(columns=[self.col])
        X_test[new_columns] = X_test_tf

        metadata["data_description"]["feature_types"].pop(self.col)
        for col in new_columns:
            if col == 'is_weekend':
                metadata["data_description"]["feature_types"][col] = "bin"
            else:
                metadata["data_description"]["feature_types"][col] = "num"

        return X_train, y_train, X_test, metadata
