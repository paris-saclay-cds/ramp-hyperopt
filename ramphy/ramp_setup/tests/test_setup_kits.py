import os
import glob
import pytest
import shutil
import hashlib
import numpy as np
import ramphy as rh
import rampwf as rw
from pathlib import Path
from ray.tune.error import TuneError

PATH = os.path.dirname(__file__)


def _generate_grid_path_kits():
    grid = []
    for path_kit in sorted(glob.glob(os.path.join(PATH, 'ramp_setup_kits', '*'))):
        grid.append(os.path.abspath(path_kit))
    return grid


@pytest.mark.parametrize(
    "path_kit",
    _generate_grid_path_kits()
)
def test_submission(path_kit):
    ramp_kit_dir = Path(PATH) / 'ramp_kits' / 'test_kit'
    ramp_data_dir = ramp_kit_dir
    ramp_kit_dir.mkdir(parents=True, exist_ok=True)

    # Setup and starting kit
    rh.kit_setup(
        download_dir = path_kit,
        ramp_kit_dir = ramp_kit_dir,
        ramp_data_dir = ramp_data_dir,
        ramp_templates_dir = None,
    )
    rh.actions.train(
        submission = 'starting_kit',
        fold_idxs = [400, 401],
        force_retrain = True,
        ramp_kit_dir = ramp_kit_dir,
        ramp_data_dir = ramp_data_dir,
    )
    
    # lgbm submission
    rh.ramp_setup.setup_scripts.tabular_regression.tabular_regression_submit(
        submission = 'lgbm',
        workflow_element_dict = {
            'regressor': 'lgbm',
            'feature_extractor': 'empty',
            'data_preprocessors': ['drop_id', 'invalid_col_names', 'cat_col_encoding',]
        },
        ramp_kit_dir = ramp_kit_dir,
        ramp_data_dir = ramp_data_dir,
        ramp_templates_dir = '/nas/ramp-hyperopt/ramphy/ramp_setup/',
    ) 
    rh.actions.train(
        submission = 'lgbm',
        fold_idxs = range(900, 903),
        force_retrain = True,
        ramp_kit_dir = ramp_kit_dir,
        ramp_data_dir = ramp_kit_dir,
    )

    # optimization
    submission = 'lgbm'
    n_trials = 9
    top_n_for_mean = 3
    n_sigma = 2
    n_folds = 7

    submissions_f_names = glob.glob(
        f'{ramp_kit_dir}/submissions/{submission}_best*')
    problem = rw.utils.assert_read_problem(ramp_kit_dir=ramp_kit_dir)
    for submission_dir in submissions_f_names:
        best_submission = submission_dir.split('/')[-1]
        hyperparameters = rh.parse_all_hyperparameters(
            submission_dir, problem.workflow
        )
        hyper_indices = [h.default_index for h in hyperparameters]
        hyper_hash = hashlib.sha256(np.ascontiguousarray(hyper_indices)).hexdigest()[:10]
        output_submission_dir =\
            f'{ramp_kit_dir}/submissions/{submission}_hyperopt_{hyper_hash}'
        print(f"{submission_dir} -> {output_submission_dir}")
        shutil.move(submission_dir, output_submission_dir)

    # hyperopt
    try:
        rh.actions.hyperopt(
            submission=submission, n_trials=n_trials, fold_idxs=range(900, 903),
            ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir, resume=True
        )
    except TuneError:
        # If there are trials that errored, Tune throws an error at the end
        pass

    # resume
    try:
        rh.actions.hyperopt(
            submission=submission, n_trials=n_trials, fold_idxs=range(900, 903),
            ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir, resume=True
        )
    except TuneError:
        # If there are trials that errored, Tune throws an error at the end
        pass
    
    rh.actions.delete_duplicates_hyperopt(
        submission=submission, fold_idxs=range(900, 903),
        ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir
    )
    
    for stop_fold_idx in range(904, 900 + n_folds + 1):
        rh.actions.select_top_hyperopt_and_train(
            submission=submission, fold_idxs=range(900, stop_fold_idx),
            trained_fold_idxs=range(900, stop_fold_idx - 1),
            top_n=top_n_for_mean, n_sigma=n_sigma,
            ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir
        )

    # This below also blends, so submissions/training_output/submission_combined_bagged_test.csv can be submitted
    rh.actions.rename_best_hyperopt_submissions(
        submission=submission, fold_idxs=range(900, 900 + n_folds), 
        top_n=10, ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir
    )
    
    rh.actions.save_hyperopt_score_summary(
        submission=submission,
        ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir
    )
    
    # Training and bagging the best on 8 folds
    submissions_f_names = glob.glob(
        f'{ramp_kit_dir}/submissions/{submission}_best_0_*')
    submissions = [f.split('/')[-1] for f in submissions_f_names]
    rh.actions.train(
        submission=submissions[0], fold_idxs=range(900, 908),
        ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir
    )

    # cleaning up
    shutil.rmtree(ramp_kit_dir)
    shutil.rmtree('cache')


