"""Actions for RAMP agent."""
import os
import time
import shutil
import hashlib
import itertools

import glob
import json

import numpy as np
import pandas as pd
import rampwf as rw
from .hyperopt import run_hyperopt, parse_hyperparameters, parse_all_hyperparameters, write_hyperparameters
from pathlib import Path

def _bagged_reward(score_type, bagged_f_name):
    bagged_scores_df = pd.read_csv(bagged_f_name)
    valid_scores_df = bagged_scores_df[bagged_scores_df['step'] == 'valid']
    r = valid_scores_df.iloc[-1][score_type.name]
    if score_type.is_lower_the_better:
        return -r
    else:
        return r

def hyperopt(submission, n_trials, fold_idxs=None, 
             ramp_kit_dir = '.', ramp_data_dir = '.',
             resume=True):
    run_hyperopt(
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
        ramp_submission_dir='submissions',
        data_label=None,
        submission=submission,
        engine_name='ray_hebo',
        n_trials=n_trials,
        fold_idxs=fold_idxs,
        save_output=True,
        test=False,
        label=False,
        resume=resume,
        max_concurrent_runs=1,
        n_cpu_per_run=8,
        n_gpu_per_run=0,
        verbose=3,
    )
    # reward TBD

def train(
        submission, fold_idxs=None, bag=True, force_retrain=False,
        ramp_kit_dir = '.', ramp_data_dir = '.'):
    """Training action. 

    Trains and bags a submission on a set of folds

    Parameters
    ----------
    submission : str
        The name of the submission to be tested.
    fold_idxs : list of int, default=None
        Fold indices to train on.
        If None, we will train on all folds.
    bag : bool, default=True
        Whether to bag the folds after training.
    force_retrain : bool, default=False
        Whether to retrain on folds with existing scores.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit.
    ramp_data_dir : str, default='.'
        The directory of the data.
    Returns
    -------
    r : float
        The reward: the bagged valid score.
    """
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    rw.utils.testing.assert_submission(
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
        submission=submission,
        save_output=True,
        retrain=False,
        bag=bag,
        force_retrain=force_retrain,
        fold_idxs=fold_idxs,
    )
    if bag:
        submission_dir = Path(ramp_kit_dir) / 'submissions' / submission
        bagged_f_name = submission_dir / 'training_output' / 'bagged_scores.csv'
        return _bagged_reward(problem.score_types[0], bagged_f_name)

def retrain(submission, ramp_kit_dir = '.', ramp_data_dir = '.'):
    """Retraining action. 

    Trains the submissin on full training data. No reward since no
    validation set.

    Parameters
    ----------
    submission : str
        The name of the submission to be tested.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit.
    ramp_data_dir : str, default='.'
        The directory of the data.
    """
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    rw.utils.testing.assert_submission(
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
        submission=submission,
        save_output=True,
        retrain=True,
        fold_idxs=[],
    )


def blend(submissions, fold_idxs=None,
          ramp_kit_dir = '.', ramp_data_dir = '.'):
    """Blending action. 

    Blends a list of submissions

    Parameters
    ----------
    submissions : list of str
        The name of the submissions to be blended.
    fold_idxs : list of int, default=None
        Fold indices to blend.
        If None, we will blend all folds.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit.
    ramp_data_dir : str, default='.'
        The directory of the data.
    Returns
    -------
    r : float
        The reward: the blended valid score.
    """
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    output_path = Path(ramp_kit_dir) / 'submissions' / 'training_output'
    rw.utils.testing.blend_submissions(
        submissions, ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir,
        ramp_submission_dir='submissions', save_output=True,
        output_path=output_path, fold_idxs=fold_idxs)

    bagged_f_name = output_path / 'bagged_scores_combined.csv'
    return _bagged_reward(problem.score_types[0], bagged_f_name)

def submit_hybrid(
        new_submission, parent_submissions,
        ramp_kit_dir = '.', ramp_data_dir = '.'):
    """Combines workflow elements coming from different submissions.
    
    Borrowing workflow elements from parent submissions.

    The number of submission must match the number of workflow elements.
    For example, in FeatureExtractorClassifierWithEDA workflow, there
    is a feature_extractor and a classifier. The new submission will 
    use the feature_extractor of the first parent and the classifier
    of the second parent.

    Parameters
    ----------
    new_submission : str
        The name of the new submission to be submitted.
    parent_submissions : list of str
        The names of the submissions from which the workflow elements
        will come.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit.
    ramp_data_dir : str, default='.'
        The directory of the data.
    """
    new_submission_dir = Path(ramp_kit_dir) / 'submissions' / new_submission
    new_submission_dir.mkdir(parents=False, exist_ok=True)
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    if len(parent_submissions) != len(problem.workflow.element_names):
        raise ValueError(f'Number of parent submissions ({len(parent_submissions)}) '\
                         f' and number of submission files ({len(problem.workflow.element_names)}) '\
                         'shuold be the same.')
        
    parents_and_wf_elements = zip(
        parent_submissions, problem.workflow.element_names)
    for parent_submission, wf_element in parents_and_wf_elements:
        from_file = Path(ramp_kit_dir) / 'submissions' / parent_submission / f'{wf_element}.py'
        to_file = Path(ramp_kit_dir) / 'submissions' / new_submission / f'{wf_element}.py'
        shutil.copy(from_file, to_file)
        print(f'Copying {from_file} to {to_file}')

def _select_all_hyperopt(
    submission, ramp_kit_dir = '.'):
    """Returns all submissions {submission}_hyperopt*.
    
    Parameters
    ----------
    submission : str
        The name of the original hyperopted submission.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit.
    Returns
    -------
    new_submissions : list of str
        The list of selected submissions.
    """
    submissions_f_names = glob.glob(
        f'{ramp_kit_dir}/submissions/{submission}_hyperopt*')

def _select_top_hyperopt(
        submission, fold_idxs, score_cutoff=None, top_n=None, 
        select_best=True,
        ramp_kit_dir = '.', ramp_data_dir = '.'):
    """Returns submissions {submission}_hyperopt* with top score.

    Selects all {submission}_hyperopt* submissions which have been
    trained on fold_idxs. Then selects either the top_n
    according to mean score, or those with mean score better than
    score_cutoff. If both are None, return all hyperopts that were trained
    on fold-idxs.
    Parameters
    ----------
    submission : str
        The name of the original hyperopted submission.
    fold_idxs : list or generator of int
        Fold indices that the {submission}_hyperopt*'s 
        have been trained on.
    score_cutoff : float, default=None
        Worst mean score of the {submission}_hyperopt*'s
        to be returned.
        Either score_cutoff or top_n must be non None.
    top_n : int, default=None
        Number of the best {submission}_hyperopt*'s
        to be returned.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit.
    ramp_data_dir : str, default='.'
        The directory of the data.
    Returns
    -------
    new_submissions : list of str
        The list of selected submissions.
    """

    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    X_train, y_train = problem.get_train_data(ramp_data_dir)
    score_names = [st.name for st in problem.score_types] 
    valid_score_name = f'valid_{score_names[0]}'
    is_lower_the_better = problem.score_types[0].is_lower_the_better
    module_path = Path(ramp_kit_dir) / 'submissions' / submission
    hypers = parse_all_hyperparameters(module_path, problem.workflow)
    hyper_names = [f'hyper_{h.name}' for h in hypers]
    hyper_index_names = [f'{hn}_i' for hn in hyper_names]
    hyper_grid_sizes = [len(h.values) for h in hypers]

    score_f_names = glob.glob(
        f'{ramp_kit_dir}/submissions/{submission}_hyperopt*/training_output/fold*/scores.csv')
    row_dicts = []
    for score_f_name in score_f_names:
        row_dict = {}
        row_dict['hyperopt_submission'] = score_f_name.split('/')[-4]
        row_dict['fold_idx'] = int(score_f_name.split('/')[-2].split('_')[1])
        hyper_module_path = Path(
            ramp_kit_dir) / 'submissions'  / row_dict['hyperopt_submission']
        hyper_hypers = parse_all_hyperparameters(
            hyper_module_path, problem.workflow)
        for h in hyper_hypers:
            row_dict[f'hyper_{h.name}_i'] = h.default_index
            row_dict[f'hyper_{h.name}'] = h.default
        score_df = pd.read_csv(
            hyper_module_path / 'training_output' / f'fold_{row_dict["fold_idx"]}' / 'scores.csv')
        score_df = score_df.set_index('step')
        for step in ['train', 'valid', 'test']:
            for sn in score_names + ['time']:
                row_dict[f'{step}_{sn}'] = score_df.loc[step, sn]
        row_dicts.append(row_dict)
    summary_df = pd.DataFrame.from_records(row_dicts)
    summary_df = summary_df[summary_df['fold_idx'].isin(fold_idxs)]
    # Select submissions that have _all_ the folds of fold_idxs trained
    if select_best:
        agg = {col:'first' if col in hyper_names else 'count'
               for col in summary_df.set_index('hyperopt_submission').columns}
        count_df = summary_df.groupby('hyperopt_submission').agg(agg)
        count_df = count_df.reset_index()
        selected_hyperopt_submissions = count_df[count_df[valid_score_name] == len(fold_idxs)]['hyperopt_submission'].to_numpy()
        summary_df = summary_df[summary_df['hyperopt_submission'].isin(selected_hyperopt_submissions)]

    agg = {col:'first' if col in hyper_names else 'mean'
           for col in summary_df.set_index('hyperopt_submission').columns}
    means_df = summary_df.groupby('hyperopt_submission').agg(agg)
    means_df[hyper_index_names] = means_df[hyper_index_names].astype(int)
    if score_cutoff is not None:
        if (is_lower_the_better and select_best) or (not is_lower_the_better and not select_best):
            new_submissions = means_df[means_df[valid_score_name] < score_cutoff].index
        else:
            new_submissions = means_df[means_df[valid_score_name] > score_cutoff].index
    elif top_n is not None:
        new_submissions = means_df.sort_values(
            valid_score_name, ascending=is_lower_the_better).iloc[:top_n].index
    else:
        new_submissions = means_df.index
    return new_submissions

def select_top_hyperopt_and_train(
        submission, fold_idxs, trained_fold_idxs=None,
        score_cutoff=None, top_n=None, 
        ramp_kit_dir='.', ramp_data_dir='.'):
    """Selects and trains submissions {submission}_hyperopt*.

    Selects all {submission}_hyperopt* submissions, then those
    which have been trained on trained_fold_idxs, if trained_fold_idxs
    is not None. Then selects either the top_n according to mean
    score, or those with mean score better than score_cutoff. Then
    trains these submissions on fold_idxs.

    If trained_fold_idxs is None, all submissions
    {submission}_hyperopt* will be trained on fold_idxs.

    For example, we have a set of {submission}_hyperopt* hyperopted
    on a smaller fold with a small number of folds, say
    trained_fold_idxs = [400, 401, 402].
    We select the top_n = 50 of them and train them on a larger number
    of larger folds fold_idxs = [700, 701, 702, 703, 704].
    Parameters
    ----------
    submission : str
        The name of the original hyperopted submission.
    fold_idxs : list or generator of int
        Fold indices to train selected submissions on.
    train_fold_idxs : list or generator of int, default=None
        Fold indices that the {submission}_hyperopt*'s 
        have been trained on. If None, all submission_hyperopt*'s
        will be trained on fold_idxs.
    score_cutoff : float, default=None
        Worst mean score of the {submission}_hyperopt*'s to be
        trained.
        If train_fold_idxs is not None, either score_cutoff or
        top_n must be non None.
    top_n : int, default=None
        Number of the best {submission}_hyperopt*'s to be
        trained.
        If train_fold_idxs is not None, either score_cutoff or
        top_n must be non None.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit.
    ramp_data_dir : str, default='.'
        The directory of the data.
    Returns
    -------
    new_submissions : list of str
        The list of selected submissions.
    """
    if trained_fold_idxs is None:
        new_submissions = _select_all_hyperopt(submission, ramp_kit_dir)
    else:
        new_submissions = _select_top_hyperopt(
            submission, trained_fold_idxs, score_cutoff, top_n, 
            ramp_kit_dir, ramp_data_dir)
    for submission in new_submissions:
        train(
            submission, fold_idxs=fold_idxs, bag=False, 
            ramp_kit_dir = '.', ramp_data_dir = '.')

def select_top_hyperopt_and_blend(
        submission, fold_idxs,
        score_cutoff=None, top_n=None, 
        ramp_kit_dir='.', ramp_data_dir='.'):
    """Selects and blends submissions {submission}_hyperopt*.

    Selects {submission}_hyperopt* submissions, then those
    which have been trained on fold_idxs, and blends them.
    Parameters
    ----------
    submission : str
        The name of the original hyperopted submission.
    fold_idxs : list or generator of int
        Fold indices that the {submission}_hyperopt*'s 
        have been trained on.
    score_cutoff : float, default=None
        Worst mean score of the {submission}_hyperopt*'s to be
        trained.
    top_n : int, default=None
        Number of the best {submission}_hyperopt*'s to be
        trained.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit.
    ramp_data_dir : str, default='.'
        The directory of the data.
    """
    submissions = _select_top_hyperopt(
        submission, fold_idxs, score_cutoff, top_n, 
        ramp_kit_dir, ramp_data_dir)
    blend(
        submissions, fold_idxs, 
        ramp_kit_dir, ramp_data_dir)

def select_top_hyperopt_and_submit_hybrid(
        new_submission, parent_submissions, select_idx,
        fold_idxs, score_cutoff=None, top_n=None, keep_hypers=False,
        ramp_kit_dir='.', ramp_data_dir='.'):
    """Selects top {submission}_hyperopt*'s and combines them with a new we.

    Selects all {parent_submissions[select_idx]}_hyperopt* submissions
    which have been trained on fold_idxs, then selects either the
    top_n according to mean score, or those with mean score better than
    score_cutoff. Then hybridizes these hyperopted submissions with the
    workflow elements of the other parent_submissions.
    
    For example, say LLM submits a new feature extractor that we pair with
    xgboost, calling it xgboost_fe. We also have 100 submissions
    xgboost_hyperopt_* with the old feature extractor, hyperopted on
    fold_idxs. We want to select the top 50 of them, and submit these 50 new
    submissions where the classifiers come from xgboost_hyperopt_*, and the
    feature extractor comes from xgboost_fe. We call this function then
    with new_submission = 'xgboost_fe' (will become the prefix of the new
    50 submissions; the hash in xgboost_fe_<hash> will be computed on the
    new submission files since the new feature extractor might have changed
    hyper values, on which the hash is computed); parent_submissions = 
    ['xgboost_fe', 'xgboost'], indicating that the feature extractor (the 
    first workflow element) will come from xgboost_fe, and the second
    (the classifier) will come from 'xgboost_hyperopt*, and select_idx=1,
    telling that it is the second parent who's hyperopted children will
    be hybridized. In case the LLMs submit the classifier, we'd have
    parent_submissions = ['xgboost', 'new_classifier'], and select_idx=0.
    
    Parameters
    ----------
    new_submission : str
        The name of the new submission to be submitted.
    parent_submissions : list of str
        The names of the submissions from which the workflow elements
        will come.
    select_idx : int
        The index of the parent submission whose (typically)
        hyperopted children will be used in the hybrid. 
    fold_idxs : list or generator of int
        Fold indices to train selected submissions on.
    score_cutoff : float, default=None
        Worst mean score of the {submission}_hyperopt*'s to be
        trained.
        Either score_cutoff or top_n must be non None.
    top_n : int, default=None
        Number of the best {submission}_hyperopt*'s to be
        trained.
    keep_hypers : bool, default=False
        If True, hypers of the top
        {parent_submissions[select_idx]}_hyperopt* will be kept
        even for the workflow elements that do not come
        from the {parent_submissions[select_idx]}_hyperopt*.
        Only works if all the parent submissions share the same
        hypers.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit.
    ramp_data_dir : str, default='.'
        The directory of the data.
    """
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    new_submissions = _select_top_hyperopt(
        parent_submissions[select_idx], fold_idxs, score_cutoff, top_n, 
        ramp_kit_dir, ramp_data_dir)
    for submission in new_submissions:
        parent_submissions[select_idx] = submission
        submit_hybrid(
            '__new_submission__', parent_submissions,
            ramp_kit_dir, ramp_data_dir)
        module_path = Path(ramp_kit_dir) / 'submissions' / '__new_submission__'
        if keep_hypers:
            orig_module_path = Path(ramp_kit_dir) / 'submissions' / submission
            hypers_per_workflow_element = {}
            for wen in problem.workflow.element_names:
                hypers_per_workflow_element[wen] = parse_hyperparameters(
                    orig_module_path, wen)
            write_hyperparameters(
                module_path, module_path, hypers_per_workflow_element)
        hypers = parse_all_hyperparameters(module_path, problem.workflow)
        hyper_indices = [h.default_index for h in hypers]
        hyper_hash = hashlib.sha256(np.ascontiguousarray(hyper_indices)).hexdigest()[:10]
        output_submission_dir =\
            Path(ramp_kit_dir) / 'submissions' / f'{new_submission}_hyperopt_{hyper_hash}'
        if not output_submission_dir.exists():  # force resubmit perhaps?
            shutil.move(module_path, output_submission_dir)

def clean_up_predictions(submission, fold_idxs, score_cutoff=None, ramp_kit_dir='.'):
    """
    
    Parameters
    ----------
    new_submission : str
        The name of the new submission to be submitted.
    parent_submissions : list of str
        The names of the submissions from which the workflow elements
        will come.
    select_idx : int
        The index of the parent submission whose (typically)
        hyperopted children will be used in the hybrid. 
    fold_idxs : list or generator of int
        Fold indices to train selected submissions on.
    score_cutoff : float, default=None
        Worst mean score of the {submission}_hyperopt*'s to be
        trained.
        Either score_cutoff or top_n must be non None.
    top_n : int, default=None
        Number of the best {submission}_hyperopt*'s to be
        trained.
        Either score_cutoff or top_n must be non None.
    keep_hypers : bool, default=False
        If True, hypers of the top
        {parent_submissions[select_idx]}_hyperopt* will be kept
        even for the workflow elements that do not come
        from the {parent_submissions[select_idx]}_hyperopt*.
        Only works if all the parent submissions share the same
        hypers.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit.
    ramp_data_dir : str, default='.'
        The directory of the data.
    """
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    if score_cutoff is None:
        submissions_f_names = glob.glob(
            f'{ramp_kit_dir}/submissions/{submission}*')
    else:
        submissions = _select_top_hyperopt(
            submission, fold_idxs, score_cutoff=score_cutoff,
            select_best=False, ramp_kit_dir=ramp_kit_dir)
        submissions_f_names = [f'submissions/{submission}' for submission in submissions]
    for module_path in submissions_f_names:
        for fold_idx in fold_idxs:
            fold_path = Path(module_path) / 'training_output' / f'fold_{fold_idx}'
            for f_name in [fold_path / 'y_pred_test.npz', fold_path / 'y_pred_train.npz']:
                if f_name.is_file():
                    print(f'Deleting {f_name}')
                    os.remove(f_name)


        
