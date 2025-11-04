import numpy as np
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from .generic_engine import GenericEngine

def prior_subset(values, priors):
    """
    Estimate subset size based on effective sample size from entropy,
    then return a random subset that size of values, sampled
    according to probability.
    Uniform distribution returns full set.
    """
    normalized_priors = [p for p in priors if p > 0]
    normalized_priors = normalized_priors / np.sum(priors)
    entropy = -np.sum(normalized_priors * np.log(normalized_priors))
    effective_size = np.exp(entropy)
    subset_size_float = round(effective_size, 3)
    fr = subset_size_float % 1
    subset_size = int(subset_size_float) + np.random.choice([0, 1], p=[1-fr, fr])
    subset = np.sort(
        np.random.choice(values, p=priors/np.sum(priors),
                         size=subset_size, replace=False)
    )
#    print(subset_size_float, subset)
    return(subset)

class HeboEngine(GenericEngine):
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
        is_lower_the_better = problem.score_types[0].is_lower_the_better
        engine_mode = "min" if is_lower_the_better else "max"
        space = DesignSpace().parse([
#            {"name": h.name, "type": "int", "lb": h.start_value_index, "ub": h.stop_value_index - 1}
            {"name": h.name, "type": "sparse_grid",
             "values": prior_subset(range(h.start_value_index, h.stop_value_index), h.priors[h.start_value_index : h.stop_value_index]),
             "lb": h.start_value_index, "ub": h.stop_value_index - 1}
            for h in self.hyperparameters
        ])
        hebo_cfg = {}
        opt = HEBO(space, **hebo_cfg)
        # constructing previous points
        valid_score_name = f"valid_{problem.score_types[0].name}"
        cols = ["hyperopt_submission", "fold_idx", valid_score_name]
        cols += [f"hyper_{h.name}_i" for h in self.hyperparameters]
        summary_df = df_scores[cols]
        prev_trials_group = summary_df.groupby("hyperopt_submission")
        mean_df = prev_trials_group.mean()
        # non-bulletproof test for submissions that have all folds trained
#        mean_df = mean_df[mean_df["fold_idx"] == np.array(hyperparameter_experiment.fold_idxs).mean()]
        mean_df = mean_df.rename(columns={f"hyper_{h.name}_i": h.name for h in self.hyperparameters})
        mean_df = mean_df.astype({h.name: 'int' for h in self.hyperparameters})

        evaluated_rewards = mean_df[valid_score_name].to_numpy().astype(float)
        points_to_evaluate = mean_df[[h.name for h in self.hyperparameters]]
        opt.observe_new_data(points_to_evaluate, evaluated_rewards)
        
        rec = opt.suggest()
        # the right order
        next_value_indices = [rec[h.name].iloc[0] for h in self.hyperparameters]
        print(next_value_indices)
        return next_value_indices

#        # Allowing priors in HEBO engine
#        next_value_indices = []
#        for h in self.hyperparameters:
#            # Normalized prior distribution
#            prior = np.clip(h.actual_priors, 0., None)
#            prior /= prior.sum()
#            selected_index = np.random.choice(range(len(prior)), p=prior)
#            next_value_indices.append(selected_index)
#        print(next_value_indices)
#        return next_value_indices

    def pass_feedback(self, df_scores, problem):
        pass
