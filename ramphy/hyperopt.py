"""Hyperparameter optimization for ramp-kits."""
import re
import os
import time
import glob
import json
import shutil
import hashlib
import itertools

import numpy as np
import pandas as pd
import rampwf as rw
from ray import tune, train
from tempfile import mkdtemp
from pathlib import Path
import warnings

# flake8: noqa: E501


from .engines import RandomEngine

HYPERPARAMS_SECTION_START = '# RAMP START HYPERPARAMETERS'
HYPERPARAMS_SECTION_END = '# RAMP END HYPERPARAMETERS'
HYPERPARAMS_REPL_REGEX = re.compile(
    '{}.*{}'.format(HYPERPARAMS_SECTION_START, HYPERPARAMS_SECTION_END), re.S
)


class Hyperparameter(object):
    """Discrete grid hyperparameter.

    Represented by a list of values, a default value, the name of the
    hyperparameter (specified by the user in the workflow element), the
    name of the workflow element in which the hyperparemeter appears, and an
    optional prior probability vector.

    Attributes:
        name : string
            The name of the hyperparameter variable, used in user interface,
            both for specifying the grid of values and getting the report on
            an experiment. Initialized to '' then set in set_names, to the
            name the user chose for the variable in the workflow element.
        workflow_element_name : string
            The name of the workflow element in which the hyperparameter is
            used. Initialized to '' then set in set_names.
        dtype : string
            The dtype of the hyperparameter.
        default_index: int
            The index in values of the current value of the hyperparameter.
        values: numpy array of any dtype
            The list of hyperparameter values.
        prior: numpy array of float
            A list of reals that the hyperopt can use as a prior probability
            over values. Positivity and summing to one are not checked,
            hyperparameter optimizers should do that when using the list
    """

    def __init__(self, dtype, default=None, values=None, prior=None):
        self.name = ''
        self.workflow_element_name = ''
        self.dtype = dtype
        if default is None and values is None:
            raise ValueError('Either default or values must be defined.')
        if values is None:
            self.values = np.array([default], dtype=self.dtype)
        else:
            if len(values) < 1:
                raise ValueError('Values needs to contain at least one element.')
            self.values = np.array(values, dtype=self.dtype)
        if default is None:
            self.default_index = 0
        else:
            if default not in self.values:
                message = 'Default must be among values.\n'
                message += f'default: {default}\n'
                message += f'values: {self.values}'
                raise ValueError(message)
            else:
                self.set_default(default)

        if prior is None:
            self.prior = np.array([1.0 / self.n_values] * self.n_values)
        else:
            if len(prior) != len(values):
                raise ValueError(
                    'len(values) == {} != {} == len(prior)'.format(
                        len(values), len(prior)
                    )
                )
            self.prior = prior

    @property
    def n_values(self):
        """The number of hyperparameter values.

        Return:
            n_values : int
                The number of hyperparameter values len(values)
        """
        return len(self.values)

    @property
    def default(self):
        """The current value of the hyperparameter.

        Return:
            default : any dtype
                The current value of the hyperparameter values[default_index].
        """
        return self.values[self.default_index]

    @property
    def default_repr(self):
        """The string representation of the default value.

        It can be used to output the default value into a python file. For
        object types it adds '', otherwise it's the string representation of
        the default value.

        Return:
            default_repr : str
                The string representation of the default value.
        """
        if self.dtype in ['object', 'str']:
            return "'{}'".format(self.default)
        else:
            return str(self.default)

    @property
    def values_repr(self):
        """The string representation of the list of values.

        It can be used to output the list of values into a python file. For
        object types it adds '' around the values, otherwise it's the list of
        string representations of the values in brackets.

        Return:
            values_repr : list of str
                The string representation of the list of values.
        """
        s = '['
        for v in self.values:
            if self.dtype in ['object', 'str']:
                s += "'{}', ".format(v)
            else:
                s += '{}, '.format(v)
        s += ']'
        return s

    @property
    def python_repr(self):
        """The string representation of the hyperparameter.

        It can be used to output the hyperparameter definition into a python
        file:
        <name> = Hyperparameter(
            dtype=<dtype>, default=<default>, values=[<values>])

        Return:
            python_repr : str
                The string representation of the hyperparameter.
        """
        repr = '{} = Hyperparameter(\n'.format(self.name)
        repr += "    dtype='{}'".format(str(self.dtype))
        repr += ', default={}'.format(self.default_repr)
        repr += ', values={})\n'.format(self.values_repr)
        return repr

    def set_names(self, name, workflow_element_name):
        """Set the name and workflow element name.

        Used when a hyperparameter object is loaded from a workflow element.

        Parameters:
            name : str
                The name of the hyperparameter, declared by the user in the
                workflow element.
            workflow_element_name : str
                The name of the workflow element in which the hyperparameter
                is defined.

        """
        self.name = name
        self.workflow_element_name = workflow_element_name

    def get_index(self, value):
        """Get the index of a value.

        Parameters:
            value : any dtype
                The value to look for.
        """
        if self.dtype == 'float':
            float_list = list([abs(v - value) < 1e-15 for v in self.values])
            return float_list.index(True)
        else:
            return list(self.values).index(value)

    def set_default(self, default):
        """Set the default value.

        Parameters:
            default : any dtype
                The new default value.
        """
        self.default_index = self.get_index(default)

    def __int__(self):
        """Cast the default value into an integer.

        It can be used in the workflow element for an integer hyperparameter.

        Return:
            int(default) : int
                The integer representation of the default value.
        """
        return int(self.default)

    def __float__(self):
        """Cast the default value into an float.

        It can be used in the workflow element for an float hyperparameter.

        Return:
            float(default) : float
                The float representation of the default value.
        """
        return float(self.default)

    def __str__(self):
        """Cast the default value into a string.

        It can be used in the workflow element for a string hyperparameter.

        Return:
            str(default) : str
                The string representation of the default value.
        """
        return str(self.default)

    def __bool__(self):
        """Cast the default value into a bool.
        It can be used in the workflow element for a bool hyperparameter.
        Return:
            bool(default) : bool
                The string representation of the default value.
        """
        return bool(self.default) 


def parse_hyperparameters(module_path, workflow_element_name):
    """Parse hyperparameters in a workflow element.

    Load the module, take all Hyperparameter objects, and set the name of each
    to the name of the hyperparameter the user chose and the workflow element
    name of each to workflow_element_name.

    Parameters:
        module_path : str
            The path to the submission directory.
        workflow_element_name : string
            The name of the workflow element.
    Return:
        hyperparameters : list of instances of Hyperparameter
    """
    hyperparameters = []
    workflow_element = rw.utils.import_module_from_source(
        os.path.join(module_path, workflow_element_name + '.py'), workflow_element_name
    )
    for object_name in dir(workflow_element):
        o = getattr(workflow_element, object_name)
        if type(o) == Hyperparameter:
            o.set_names(object_name, workflow_element_name)
            hyperparameters.append(o)
    return hyperparameters


def parse_all_hyperparameters(module_path, workflow):
    """Parse hyperparameters in a submission.

    Load all the the modules, take all Hyperparameter objects, and set the name
    of each to the name of the hyperparameter the user chose and the workflow
    element name of each to the corresponding workflow_element_name.

    Parameters:
        module_path : str
            The path to the submission directory.
        workflow : workflow from rw.workflows 
            The ramp workflow.
    Return:
        hyperparameters : list of instances of Hyperparameter
    """
    hyperparameters = []
    for wen in workflow.element_names:
        hyperparameters += parse_hyperparameters(module_path, wen)
    return hyperparameters


def write_hyperparameters(
    submission_dir, output_submission_dir, hypers_per_workflow_element
):
    """Write hyperparameters in a submission.

    Read workflow elements from submission_dir, replace the hyperparameter
    section with the hyperparameters in the hypers_per_workflow_element
    dictionary (with new hyperparamter values set by, e.g, a hyperopt engine),
    then write the new workflow elements into output_submission_dir (which
    can be a temporary directory or submission_dir itself when the function
    is called to replace the hyperparameters in the input submission with the
    best hyperparameters.)

    Parameters:
        submission_dir : str
            The path to the submission directory from which the submission is
            read.
        output_submission_dir : str
            The path to the output submission directory into which the
            submission with the new hyperparameter values is written.
        hypers_per_workflow_element : dictionary
            Each key is a workflow element name and each value is a list of
            Hyperparameter instances, representing the hyperparemters in
            the workflow element.
    """
    for wen, hs in hypers_per_workflow_element.items():
        hyper_section = '{}\n'.format(HYPERPARAMS_SECTION_START)
        for h in hs:
            hyper_section += h.python_repr
        hyper_section += HYPERPARAMS_SECTION_END
        f_name = os.path.join(submission_dir, wen + '.py')
        with open(f_name) as f:
            content = f.read()
            content = HYPERPARAMS_REPL_REGEX.sub(hyper_section, content)
        Path(output_submission_dir).mkdir(parents=True, exist_ok=True)
        output_f_name = Path(output_submission_dir) / f'{wen}.py'
        with open(output_f_name, 'w') as f:
            f.write(content)


class HyperparameterOptimization(object):
    """A hyperparameter optimization.

    Attributes:
        hyperparameters: a list of Hyperparameters
        engine: a hyperopt engine
        ramp_kit_dir: the directory where the ramp kit is found
        submission_dir: the directory where the submission to be optimized
            is found
    """

    def __init__(
        self, hyperparameters, engine, ramp_kit_dir, ramp_data_dir,
        submission_dir, fold_idxs, data_label, test
    ):
        self.hyperparameters = hyperparameters
        self.engine = engine
        self.problem = rw.utils.assert_read_problem(ramp_kit_dir)
        if data_label is not None:
            self.X_train, self.y_train = self.problem.get_train_data(
                path=ramp_kit_dir, data_label=data_label
            )
            self.X_test, self.y_test = self.problem.get_test_data(
                path=ramp_kit_dir, data_label=data_label
            )
        else:
            self.X_train, self.y_train = self.problem.get_train_data(path=ramp_data_dir)
            self.X_test, self.y_test = self.problem.get_test_data(path=ramp_data_dir)
        cv_gen = self.problem.get_cv(self.X_train, self.y_train)
        self.cv = []
        if fold_idxs is None:
            fold_start = 0
            fold_stop = None
        else:
            fold_start = min(fold_idxs)
            fold_stop = max(fold_idxs) + 1
        fold_i = fold_start - 1
        for fold in itertools.islice(cv_gen, fold_start, fold_stop):
            fold_i += 1
            if fold_idxs is None or fold_i in fold_idxs:
                self.cv.append(fold)
        if fold_idxs is None:
            self.fold_idxs = list(range(0, fold_i + 1))
        else:
            self.fold_idxs = fold_idxs

        self.data_label = data_label
        self.submission_dir = submission_dir
        self.test = test

        self.hyperopt_output_path = os.path.join(
            self.submission_dir, 'hyperopt_output')
        if self.data_label is not None:
            self.hyperopt_output_path = os.path.join(
                self.hyperopt_output_path, self.data_label)
        if not os.path.exists(self.hyperopt_output_path):
            os.makedirs(self.hyperopt_output_path)

        self.hyperparameter_names = [h.name for h in hyperparameters]
        self.hyperparameters_indices = [h.name + '_i' for h in hyperparameters]
        self.score_names = [s.name for s in self.problem.score_types]
        self.df_summary_ = None
        self.fold_i = 0

        # Set up hypers_per_workflow_element dictionary: keys are
        # workflow element names, values are lists are hypers belonging
        # to the workflow element
        self.hypers_per_workflow_element = {
            wen: [] for wen in self.problem.workflow.element_names
        }
        for h in self.hyperparameters:
            self.hypers_per_workflow_element[h.workflow_element_name].append(h)

        # Set up df_scores_ which will contain one row per experiment
        scores_columns = ['fold_i']
        scores_columns += self.hyperparameter_names
        scores_columns += self.hyperparameters_indices
        scores_columns += ['train_' + name for name in self.score_names]
        scores_columns += ['valid_' + name for name in self.score_names]
        scores_columns += ['train_time', 'valid_time', 'n_train', 'n_valid']
        dtypes = (
            ['int']
            + [h.dtype for h in self.hyperparameters]
            + ['int'] * len(self.hyperparameters)
            + ['float'] * 2 * len(self.score_names)
            + ['float'] * 2
            + ['int'] * 2
        )
        if self.test:
            scores_columns += ['test_' + name for name in self.score_names]
            scores_columns += ['test_time', 'n_test']
            dtypes = dtypes + (
                ['float'] * len(self.score_names) + ['float'] + ['int'])

        self.df_scores_ = pd.DataFrame(columns=scores_columns)
        for column, dtype in zip(scores_columns, dtypes):
            self.df_scores_[column] = self.df_scores_[column].astype(dtype)

    def update_df_scores(self, df_scores, fold_i):
        row = {'fold_i': fold_i}
        for h in self.hyperparameters:
            row[h.name] = h.default
            row[h.name + '_i'] = h.default_index
        for name in self.score_names:
            row['train_' + name] = df_scores.loc['train'][name]
            row['valid_' + name] = df_scores.loc['valid'][name]
            if self.test:
                row['test_' + name] = df_scores.loc['test'][name]
        row['train_time'] = float(df_scores.loc['train']['time'])
        row['valid_time'] = float(df_scores.loc['valid']['time'])
        row['n_train'] = len(self.cv[fold_i][0])
        row['n_valid'] = len(self.cv[fold_i][1])
        if self.test:
            row['test_time'] = float(df_scores.loc['test']['time'])
            row['n_test'] = len(self.X_test)

        self.df_scores_.loc[len(self.df_scores_)] = pd.Series(row)
        self.df_scores_['fold_i'] = self.df_scores_['fold_i'].astype(int)
        for h in self.hyperparameters:
            col = h.name + '_i'
            self.df_scores_[col] = self.df_scores_[col].astype(int)

    def run_next_experiment(self, module_path, fold_i, save_output=False):
        if save_output:
            training_output_path = Path(module_path) / 'training_output'
            training_output_path.mkdir(parents=True, exist_ok=True)
            print('Training output path: {}'.format(training_output_path))
            fold_output_path = training_output_path / f'fold_{self.fold_idxs[fold_i]}'
            fold_output_path.mkdir(parents=True, exist_ok=True)
        else:
            fold_output_path='.'
        _, _, df_scores = rw.utils.run_submission_on_cv_fold(
            self.problem,
            module_path=module_path,
            fold=self.cv[fold_i],
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            save_output=save_output,
            fold_output_path=fold_output_path,
        )
        return df_scores

    def make_and_save_summary(self, path=None, fname='summary.csv'):
        if path is None:
            path = self.hyperopt_output_path
        summary_fname = Path(path) / fname
        self.df_scores_.to_csv(summary_fname)

    def load_summary(self, path=None, fname='summary.csv'):
        if path is None:
            path = self.hyperopt_output_path
        summary_fname = Path(path) / fname
        self.df_scores_ = pd.read_csv(summary_fname, index_col=0)

    def save_best_model(self):
        official_scores = self.df_summary_[
            'valid_' + self.problem.score_types[0].name + '_m'
        ]
        print('official scores', official_scores)
        if self.problem.score_types[0].is_lower_the_better:
            best_defaults = official_scores.idxmin()
        else:
            best_defaults = official_scores.idxmax()
        print('Best hyperparameters: ', best_defaults)
        try:
            for bd, h in zip(best_defaults, self.hyperparameters):
                h.set_default(bd)
        except (TypeError):
            # single hyperparameter
            self.hyperparameters[0].set_default(best_defaults)
        # Overwrite the submission with the best hyperparameter values
        write_hyperparameters(
            self.submission_dir, self.submission_dir, self.hypers_per_workflow_element
        )


def run(hyperparameter_experiment, n_trials, resume=False):
    start_iter = 0
    if resume:
        hyperparameter_experiment.load_summary(
            hyperparameter_experiment.hyperopt_output_path)
        start_iter = len(hyperparameter_experiment.df_scores_)
    start = pd.Timestamp.now()
    for i_iter in range(start_iter, n_trials):
        # Getting new hyperparameter values from engine
        (
            fold_i,
            next_value_indices,
        ) = hyperparameter_experiment.engine.next_hyperparameter_indices(
            hyperparameter_experiment.df_scores_,
            len(hyperparameter_experiment.cv),
            hyperparameter_experiment.problem,
        )
        # Updating hyperparameters
        for h, i in zip(hyperparameter_experiment.hyperparameters, next_value_indices):
            h.default_index = i
        # Writing submission files with new hyperparameter values
        output_submission_dir = mkdtemp()
        write_hyperparameters(
            hyperparameter_experiment.submission_dir,
            output_submission_dir,
            hyperparameter_experiment.hypers_per_workflow_element,
        )
        # Calling the training script.

        df_scores = hyperparameter_experiment.run_next_experiment(
            output_submission_dir, fold_i
        )
        sn = hyperparameter_experiment.score_names[0]
        hyperparameter_experiment.engine.pass_feedback(
            fold_i, len(hyperparameter_experiment.cv), df_scores, sn
        )
        hyperparameter_experiment.update_df_scores(df_scores, fold_i)
        shutil.rmtree(output_submission_dir)
        now = pd.Timestamp.now()
        eta = start + (now - start) / (i_iter + 1 - start_iter) * (
            n_trials - start_iter
        )
        print(f'Done {i_iter + 1} / {n_trials} at {now}. ETA = {eta}.')
        hyperparameter_experiment.make_and_save_summary()
    scores_columns = ['valid_' + name for name in hyperparameter_experiment.score_names]
    for score in scores_columns:
        hyperparameter_experiment.df_scores_[score + '_max'] = (
            hyperparameter_experiment.df_scores_[score]
            .rolling(n_trials, min_periods=1)
            .max()
        )


def objective(config, run_params=None):
    hyperparam_opt = run_params['hyperparam_opt']
    for h in hyperparam_opt.hyperparameters:
        h.default_index = config[h.name]
    hyper_indices = [h.default_index for h in hyperparam_opt.hyperparameters]
    hyper_hash = hashlib.sha256(np.ascontiguousarray(hyper_indices)).hexdigest()[:10]
    output_submission_dir =\
        f'{hyperparam_opt.submission_dir}_hyperopt_{hyper_hash}'
    os.chdir(run_params['current_dir'])
    write_hyperparameters(
        hyperparam_opt.submission_dir,
        output_submission_dir,
        hyperparam_opt.hypers_per_workflow_element,
    )
    # Calling the training script.
    valid_scores = np.zeros(len(hyperparam_opt.cv))
    df_scores_list = []
    for fold_i in range(len(hyperparam_opt.cv)):
        df_scores = hyperparam_opt.run_next_experiment(
            output_submission_dir, fold_i, run_params['save_output'])
        sn = hyperparam_opt.score_names[0]
        valid_scores[fold_i] = df_scores.loc['valid', sn]
        df_scores_list.append(df_scores)
        hyperparam_opt.update_df_scores(
            df_scores,
            fold_i,
        )
#    fname = f'tmp_summary_{time.time()}.csv'
#    fname = Path(output_submission_dir) / 'summary.csv'
    hyperparam_opt.make_and_save_summary(path=output_submission_dir)
#    shutil.rmtree(output_submission_dir)

    train.report({
        'valid_score': valid_scores.mean(),
        'df_scores_list': df_scores_list,
    })


def run_tune(
    hyperparameter_experiment,
    n_trials,
    max_concurrent_runs,
    n_cpu_per_run,
    n_gpu_per_run,
    save_output,
    verbose,
):

    is_lower_the_better = hyperparameter_experiment.problem.score_types[
        0
    ].is_lower_the_better
    engine_mode = 'min' if is_lower_the_better else 'max'

    if hyperparameter_experiment.engine.name == 'ray_grid_search':
        warnings.warn("A full grid search is being used with RAY's default engine !")
        config = {
            h.name: tune.grid_search([i for i in range(len(h.values))])
            for h in hyperparameter_experiment.hyperparameters
        }
        num_samples = int(len(hyperparameter_experiment.cv))
    else:
        config = {
            h.name: tune.randint(0, len(h.values))
            for h in hyperparameter_experiment.hyperparameters
        }
        num_samples = int(n_trials / len(hyperparameter_experiment.cv))

    run_params = {
        'current_dir': os.getcwd(),
        'hyperparam_opt': hyperparameter_experiment,
        'save_output': save_output,
    }
    tune_name = f'{hyperparameter_experiment.engine.name}__' +\
                f'{hyperparameter_experiment.submission_dir.split("/")[-1]}__' +\
                f'{hyperparameter_experiment.data_label}'
    results = tune.run(
        tune.with_parameters(objective, run_params=run_params),
        max_concurrent_trials=max_concurrent_runs,
        metric='valid_score',
        mode=engine_mode,
        num_samples=num_samples,
        name=tune_name,
        search_alg=hyperparameter_experiment.engine.ray_engine,
        config=config,
        verbose=verbose,
        resources_per_trial={"cpu": n_cpu_per_run, "gpu": n_gpu_per_run},
#        local_dir=f"./{(Path(hyperparameter_experiment.hyperopt_output_path) / 'ray_results').as_posix()}",
    )

    for _, row in results.results_df.iterrows():
        for h in hyperparameter_experiment.hyperparameters:
            h.default_index = int(row[f'config/{h.name}'])
        for fold_i, df_scores in enumerate(row['df_scores_list']):
            hyperparameter_experiment.update_df_scores(
                df_scores,
                fold_i,
            )

    hyperparameter_experiment.make_and_save_summary()
    results.results_df.to_csv(os.path.join(Path(hyperparameter_experiment.hyperopt_output_path), 'ray_summary.csv'))


class RayEngine:
    # n_trials is only needed by zoopt at init time
    def __init__(self, engine_name, n_trials=None, points_to_evaluate=None, evaluated_rewards=None):
        self.name = engine_name
        if (engine_name[4:] == 'random') or (engine_name[4:] == 'grid_search'):
            self.ray_engine = None
        elif engine_name[4:] == 'zoopt':
            try:
                from ray.tune.search.zoopt import ZOOptSearch

                self.ray_engine = ZOOptSearch(  # gets stuck often
                    algo='Asracos',  # only support ASRacos currently
                    budget=n_trials,
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except('zoopt')
        elif engine_name[4:] == 'ax':
            try:
                from ray.tune.search.ax import AxSearch

                self.ray_engine = AxSearch(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except('ax-platform sqlalchemy')
        elif engine_name[4:] == 'blend_search':
            try:
                from ray.tune.search.flaml import BlendSearch

                self.ray_engine = BlendSearch(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except('flaml')
        elif engine_name[4:] == 'cfo':
            try:
                from ray.tune.search.flaml import CFO

                self.ray_engine = CFO(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except('flaml')
        elif engine_name[4:] == 'skopt':
            try:
                from ray.tune.search.skopt import SkOptSearch

                self.ray_engine = SkOptSearch(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except('scikit-optimize')
        elif engine_name[4:] == 'hyperopt':
            try:
                from ray.tune.search.hyperopt import HyperOptSearch

                self.ray_engine = HyperOptSearch(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except('hyperopt')
        elif engine_name[4:] == 'bayesopt':
            try:
                from ray.tune.search.bayesopt import BayesOptSearch

                self.ray_engine = BayesOptSearch(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except('bayesian-optimization')
        elif engine_name[4:] == 'bohb':
            try:
                from ray.tune.search.bohb import TuneBOHB

                self.ray_engine = TuneBOHB(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except('hpbandster')
        elif engine_name[4:] == 'nevergrad':
            try:
                from ray.tune.search.nevergrad import NevergradSearch
                import nevergrad as ng

                self.ray_engine = NevergradSearch(
                    optimizer=ng.optimizers.OnePlusOne,
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except('nevergrad')
        elif engine_name[4:] == 'hebo':
            try:
                from ray.tune.search.hebo import HEBOSearch

                self.ray_engine = HEBOSearch(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except('hebo')
        elif engine_name[4:] == 'optuna':
            try:
                from ray.tune.search.optuna import OptunaSearch

                self.ray_engine = OptunaSearch(
                    points_to_evaluate=points_to_evaluate,
                    evaluated_rewards=evaluated_rewards,
                )
            except ModuleNotFoundError:
                self.raise_except('optuna')
        else:
            raise ValueError(f'Engine {engine_name[4:]} not found in Ray Tune')

    def raise_except(library):
        raise EnvironmentError(
            'Missing module: install it using pip install ' + library
        )


def init_hyperopt(
    ramp_kit_dir,
    ramp_data_dir,
    ramp_submission_dir,
    submission,
    engine_name,
    fold_idxs,
    data_label,
    label,
    resume,
    test,
    n_trials=None,
):
    # n_trials is only needed by ray_zoopt at init time
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    submission_dir = os.path.join(ramp_submission_dir, submission)
    hyperparameters = parse_all_hyperparameters(
        submission_dir, problem.workflow
    )
    if engine_name == 'random':
        engine = RandomEngine(hyperparameters)
    elif engine_name.startswith('ray_'):
        points_to_evaluate = None
        evaluated_rewards = None
        if resume:
            previous_trial_paths = glob.glob(f'{ramp_submission_dir}/{submission}_hyperopt_*')
            print("\n-------------- Resuming previous trials --------------")
            points_to_evaluate = []
            evaluated_rewards = []
            for prev_trial_path in previous_trial_paths:
                scores = []
                for fold_idx in fold_idxs: 
                    try:
                        score = rw.utils.load_submission_fold_score(
                            Path(prev_trial_path), fold_idx, problem.score_types[0].name,
                            'valid', data_label)
                        scores.append(score)
                    except FileNotFoundError:
                        print(f"{prev_trial_path}/fold_{fold_idx}' doesn't exist.")
                        break
                if len(scores) != len(fold_idxs):
                    print(f"Skipping {prev_trial_path}")
                    # We may later figure out how to combine previous results with
                    # heterogeneous uneven folds.
                    continue
                trial_mean_score = np.array(scores).mean()
                hyperparameters_trial = parse_all_hyperparameters(
                    prev_trial_path, problem.workflow
                )
                h_names = [h.name for h in hyperparameters_trial]
                trial_hypers = {
                    h.name: h.default_index
                    for h in hyperparameters_trial
                }
                points_to_evaluate.append(trial_hypers)
                evaluated_rewards.append(trial_mean_score)
            print(f"Found {len(points_to_evaluate)} existing subissions, resuming.") 
            print("-------------- Done --------------\n")
        engine = RayEngine(engine_name, n_trials, points_to_evaluate, evaluated_rewards)
    else:
        raise ValueError(f'{engine_name} is not a valid engine name')
    hyperparameter_experiment = HyperparameterOptimization(
        hyperparameters,
        engine,
        ramp_kit_dir,
        ramp_data_dir,
        submission_dir,
        fold_idxs,
        data_label,
        test,
    )

    return hyperparameter_experiment


def run_hyperopt(
    ramp_kit_dir,
    ramp_data_dir,
    ramp_submission_dir,
    data_label,
    submission,
    engine_name,
    n_trials,
    fold_idxs,
    save_output,
    test,
    label,
    resume,
    max_concurrent_runs,
    n_cpu_per_run,
    n_gpu_per_run,
    verbose,
):
    hyperparameter_experiment = init_hyperopt(
        ramp_kit_dir,
        ramp_data_dir,
        ramp_submission_dir,
        submission,
        engine_name,
        fold_idxs,
        data_label,
        label,
        resume,
        test,
    )
    if engine_name.startswith('ray_'):
        run_tune(
            hyperparameter_experiment,
            n_trials,
            max_concurrent_runs,
            n_cpu_per_run,
            n_gpu_per_run,
            save_output,
            verbose,
        )
    else:
        run(hyperparameter_experiment, n_trials, resume)
