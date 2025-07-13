import numpy as np
from .generic_engine import GenericEngine


class RandomEngine(GenericEngine):
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def next_hyperparameter_indices(self, df_scores, problem):
        """Return the next hyperparameter indices to try.

        Parameters:
            df_scores : pandas DataFrame
                It represents the results of the experiments that have been
                run so far.
        Return:
            next_value_indices : list of int
                The indices corresponding to the values lists in
                hyperparameters.
        """
        next_value_indices = []
        for h in self.hyperparameters:
            # Normalized prior distribution
            prior = np.clip(h.actual_priors, 0., None)
            prior /= prior.sum()
            selected_index = np.random.choice(range(len(prior)), p=prior)
            next_value_indices.append(selected_index)
        return next_value_indices

    def pass_feedback(self, df_scores, problem):
        pass
