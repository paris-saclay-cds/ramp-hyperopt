"""Actions for RAMP agent."""
import os
import shutil
import time

import glob
import json

import numpy as np
import pandas as pd
import rampwf as rw
from pathlib import Path

def _bagged_reward(score_type, bagged_f_name):
    bagged_scores_df = pd.read_csv(bagged_f_name)
    valid_scores_df = bagged_scores_df[bagged_scores_df['step'] == 'valid']
    r = valid_scores_df.iloc[-1][score_type.name]
    if score_type.is_lower_the_better:
        return -r
    else:
        return r

def train(submission, fold_idxs=None, ramp_kit_dir = '.', 
          ramp_data_dir = '.'):
    """Training action. 

    Trains and bags a submission on a set of folds

    Parameters
    ----------
    submission : str
        The name of the submission to be tested.
    fold_idxs : list of int, default=None
        The list of CV folds we want to run the submission on.
        If None, we will run on all folds.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit to be tested for submission.
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
        fold_idxs=fold_idxs,
    )
    submission_dir = Path(ramp_kit_dir) / 'submissions' / submission
    bagged_f_name = submission_dir / 'training_output' / 'bagged_scores.csv'
    return _bagged_reward(problem.score_types[0], bagged_f_name)

def bag(submission, fold_idxs=None, ramp_kit_dir = '.', ramp_data_dir = '.',
        score_f_name_prefix=''):
    """Bagging action. 

    Bags a submission on a set of folds

    Parameters
    ----------
    submission : str
        The name of the submission to be tested.
    fold_idxs : list of int, default=None
        The list of CV folds we want to run the submission on.
        If None, we will run on all folds.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit to be tested for submission.
    ramp_data_dir : str, default='.'
        The directory of the data.
    score_f_name_prefix : str, default=''
        The suffix we add to mark the submission file:
        submission_{score_f_name_prefix}bagged_test.csv
    Returns
    -------
    r : float
        The reward: the bagged valid score.
    """
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    submission_dir = Path(ramp_kit_dir) / 'submissions' / submission
    print(f'Bagging {submission} on {problem.problem_title}')
    X_train, y_train, X_test, y_test = rw.utils.assert_data(
        ramp_kit_dir, ramp_data_dir)
    cv = rw.utils.assert_cv(ramp_kit_dir, ramp_data_dir)
    if fold_idxs is None:
        fold_idxs = range(len(cv))
    training_output_path = submission_dir / 'training_output'
    print(f'Training output path: {training_output_path}')

    # saving predictions for CV bagging after the CV loop
    predictions_valid_list = []
    predictions_test_list = []

    print(f'Reading prediction files ...')
    for fold_i in fold_idxs:
        _, valid_is = cv[fold_i]
        fold_output_path = training_output_path / f'fold_{fold_i}'
        predictions_valid, predictions_test = rw.utils.load_predictions(
            problem, valid_is, data_path=ramp_data_dir,
            input_path=fold_output_path)
        predictions_valid_list.append(predictions_valid)
        predictions_test_list.append(predictions_test)

    rw.utils.bag_submissions(
        problem, cv, y_train, y_test, predictions_valid_list,
        predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=None,
        save_output=True, fold_idxs=fold_idxs,
        score_f_name_prefix=score_f_name_prefix)

    bagged_f_name = submission_dir / 'training_output' / 'bagged_scores.csv'
    return _bagged_reward(problem.score_types[0], bagged_f_name)

def blend(submissions, fold_idxs=None, ramp_kit_dir = '.', 
          ramp_data_dir = '.'):
    """Blending action. 

    Blends a list of submissions

    Parameters
    ----------
    submissions : list of str
        The name of the submissions to be blended.
    fold_idxs : list of int, default=None
        The list of CV folds we want to run the submission on.
        If None, we will run on all folds.
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit to be tested for submission.
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
        ramp_submission_dir='submissions', save_output=True, output_path=output_path)

    bagged_f_name = output_path / 'bagged_scores_combined.csv'
    return _bagged_reward(problem.score_types[0], bagged_f_name)

def submit_hybrid(new_submission, parent_submissions,
                  ramp_kit_dir = '.', ramp_data_dir = '.'):
    """Hybrid submission action.
    
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
        The directory of the ramp-kit to be tested for submission.
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

    