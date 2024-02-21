"""Hyperparameter optiomization for ramp-kits."""
import os
import shutil
import time

import glob
import json

import numpy as np
import pandas as pd
import rampwf as rw
from pathlib import Path


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
    # reward is the valid bagged score 
    submission_dir = Path(ramp_kit_dir) / 'submissions' / submission
    bagged_scores_df = pd.read_csv(
        submission_dir / 'training_output' / 'bagged_scores.csv', index_col=0)
    r = bagged_scores_df.loc['valid'].iloc[-1][problem.score_types[0].name]
    if problem.score_types[0].is_lower_the_better:
        return -r
    else:
        return r


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
    # reward is the valid bagged score 
    bagged_scores_df = pd.read_csv(
        output_path / 'bagged_scores_combined.csv', index_col=0)
    r = bagged_scores_df.loc['valid'].iloc[-1][problem.score_types[0].name]
    if problem.score_types[0].is_lower_the_better:
        return -r
    else:
        return r


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

    