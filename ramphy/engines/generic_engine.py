from abc import ABC, abstractmethod


class GenericEngine(ABC):
    def __init__(self, hyperparameters):
        pass

    @abstractmethod
    def next_hyperparameter_indices(self, df_scores, problem):
        pass

    @abstractmethod
    def pass_feedback(self, df_scores, problem):
        pass
