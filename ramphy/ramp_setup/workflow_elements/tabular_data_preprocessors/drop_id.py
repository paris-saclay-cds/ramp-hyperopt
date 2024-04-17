import regex

class DataPreprocessor(object):
    def preprocess(self, X_train, y_train, X_test, metadata):
        X_train = X_train.drop(columns=[metadata['id_name']])
        X_test = X_test.drop(columns=[metadata['id_name']])
        metadata['data_description']['feature_types'].pop(metadata['id_name'])
        return X_train, y_train, X_test, metadata