import regex

class DataPreprocessor(object):
    def preprocess(self, X_train, y_train, X_test, metadata):
        X_train = X_train.drop(columns=[metadata['id_col']])
        X_test = X_test.drop(columns=[metadata['id_col']])
        metadata['col_types'].pop(metadata['id_col'])
        return X_train, y_train, X_test, metadata