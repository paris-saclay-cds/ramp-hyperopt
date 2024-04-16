class FeatureExtractor():
    def __init__(self, metadata):
        self.metadata = metadata

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X