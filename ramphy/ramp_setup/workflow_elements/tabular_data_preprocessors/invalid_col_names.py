import re

class DataPreprocessor(object):
    def preprocess(self, X_train, y_train, X_test, metadata):
        new_cols = {{col: re.sub(r'[^A-Za-z0-9_]+', '', col)
                    for col in metadata['col_types']}}
        X_train = X_train.rename(columns=new_cols)
        X_test = X_test.rename(columns=new_cols)
        metadata['col_types'] = {{
            new_cols[col]: col_type for col, col_type in metadata['col_types'].items()}}
        return X_train, y_train, X_test, metadata