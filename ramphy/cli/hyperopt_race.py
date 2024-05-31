import glob
import json
import shutil
import random
import datetime
import numpy as np
import pandas as pd
import rampwf as rw
import ramphy as rh
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
rh.actions.EXECUTE_PLAN = True
import click
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


# flake8: noqa: E501

def last_action(ramp_kit_dir, name):
    """Last action of a given action name."""
    action_f_names = glob.glob(f'{ramp_kit_dir}/actions/*')
    action_f_names.sort(reverse=True)
    for i in range(len(action_f_names)):
        ramp_action_object = rh.actions.load_ramp_action(action_f_names[i])
        if ramp_action_object.name == name:
            return ramp_action_object
    return None

@rh.actions.ramp_action
def kaggle_submit_file(submission_source_f_name, submission_target_f_name, ramp_kit_dir):
    shutil.copy(submission_source_f_name, submission_target_f_name)

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    help="The kit to hyperopt.",
)
@click.option(
    "--version",
    help="The program version",
)
@click.option(
    "--number",
    help="The program number (repeated within version)",
)
@click.option(
    "--resume",
    is_flag=True,
    show_default=True,
    help="If True, resume from a previous broken hyperopt run",
)
def main(
    ramp_kit,
    version,
    number,
    resume,
):
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = f"{ramp_kit}_{kit_suffix}"

    submissions = ['lgbm', 'xgboost', 'catboost']
    n_rounds = 100
    n_trials_per_round = 15  # 3 folds x 5 trials
    # when selecting the top submissions within n_sigma from the best, top_n_for_mean is used to compute
    # the best and sigma ((average of top_n_for_mean best)
    top_n_for_mean = 10
    n_sigma = 1
    n_folds = 31  # to construct final blends
    contributivity_floor = 100  # on 1000, added to contributivity to give a chance to every submission
    
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    score_names = [st.name for st in problem.score_types] 
    valid_score_name = f'valid_{score_names[0]}'
    is_lower_the_better = problem.score_types[0].is_lower_the_better
    with open(Path(ramp_kit_dir) / "data" / "metadata.json", "r") as f:
        metadata = json.load(f)
    kaggle_submissions_path = Path(ramp_kit_dir) / "kaggle_submissions"

    action_stats = {submission: [] for submission in submissions}
    improvement_speed_df = pd.DataFrame(columns=["round"] + submissions)
    if resume:
        print("Loading actions...")
        action_f_names = glob.glob(f'{ramp_kit_dir}/actions/*')
        action_f_names.sort()
        ramp_program = []
        for action_f_name in action_f_names:
            f_name = Path(action_f_name).name
            ramp_program.append(rh.actions.load_ramp_action(action_f_name))
        blend_actions = [ra for ra in ramp_program if ra.name == "blend"]
        stop_time = blend_actions[-1].stop_time
        print(f"Last blending action at {stop_time}, deleting all actions after...")
        actions_f_names_to_delete = [a for a in action_f_names if pd.to_datetime(Path(a).stem) > stop_time]
        for f_name in actions_f_names_to_delete:
            Path(f_name).unlink()
        print("Re-loading actions...")
        action_f_names = glob.glob(f'{ramp_kit_dir}/actions/*')
        action_f_names.sort()
        ramp_program = []
        for action_f_name in action_f_names:
            f_name = Path(action_f_name).name
            ramp_program.append(rh.actions.load_ramp_action(action_f_name))
        blend_actions = [ra for ra in ramp_program if ra.name == "blend"]
        hyperopt_actions = [ra for ra in ramp_program if ra.name == "hyperopt"]
        n_rounds -= len(hyperopt_actions)
        print(f"Recovering {len(hyperopt_actions)} hyperopt and blend actions...")
        blend_action_idx = 0
        for hyperopt_action in hyperopt_actions:
            submission = hyperopt_action.kwargs["submission"]
            if len(hyperopt_action.mean_scores) > 0:
                blend_action = blend_actions[blend_action_idx]
                blend_action_idx += 1  # if hyperopt did not return with any submissions, there was no blend
                blended_score = blend_action.blended_score
                contributivities = {
                    s: contributivity_floor + np.array([c for sh, c in blend_action.contributivities.items() if sh[:len(s)] == s]).sum()
                    for s in submissions
                }
                action_stats[submission].append({
                    "runtime": hyperopt_action.runtime,
                    "contributivities": contributivities,
                })
                blended_submissions = set([sh for sh, c in blend_action.contributivities.items() if c > 0])
        print(f'Blended submissions: {blended_submissions}')
    else:
        for submission in submissions:
            rh.ramp_setup.setup_scripts.tabular_regression.tabular_regression_columnwise_first_submit(         
                ramp_kit_dir = ramp_kit_dir,
                submission = submission,
                regressor = submission,
            )                                                   
        kaggle_submissions_path.mkdir(parents=False, exist_ok=True)
        #Dictionary of submissions: list of dictionary of run times and scores
        blended_submissions = set()    
    
    for round_idx in range(n_rounds):
        # We'll choose the submission that improves the results the fastest
        improvement_speeds = {}
        for submission in submissions:
            ast = action_stats[submission]
            # sum of improvements of the blended score, applying this action
            if len(ast) >= 2:
                last_ast = ast[-1]
                total_time = np.array([a["runtime"].total_seconds() for a in ast]).sum()
                contributivity = last_ast["contributivities"][submission]
                speed = contributivity / total_time
            else:
                speed = np.finfo(float).max            
            improvement_speeds[submission] = speed
        print(f'improvement_speeds:\n{improvement_speeds}')
        row = pd.DataFrame(dict({"round_idx" : round_idx}, **improvement_speeds), index=[round_idx])
        improvement_speed_df = pd.concat([improvement_speed_df, row], ignore_index=True)
        improvement_speed_df.to_csv("improvement_speeds.csv")
        max_speed = max(improvement_speeds.values())
        best_submissions = [submission for submission, speed in improvement_speeds.items() if speed == max_speed]
        submission = random.choice(best_submissions)  # in case of tie (at zero typically), or eps greedy, random choice
        print(f'best_submissions: {best_submissions}')
        print(f'selected submission : {submission}')
    #    input("Press Enter to continue...")
        n_trials = n_trials_per_round
        rh.actions.hyperopt(
            ramp_kit_dir = ramp_kit_dir,
            submission = submission,
            workflow_element_names = ['regressor'],
            n_trials=n_trials,
            fold_idxs=range(900, 903),
            resume=True,
            subtract_existing=False,
        )
        hyperopt_action = last_action(ramp_kit_dir, "hyperopt")
        if len(hyperopt_action.mean_scores) > 0:
            # Add the best submission from this round of hyperopt to the set to be blended
            for hyperopt_submission, mean_score in hyperopt_action.mean_scores.items():
                # Add best
                #if mean_score == hyperopt_action.mean_score:
                blended_submissions.add(hyperopt_submission)
        
            # The blended score improvement is wrt the previous blended score. If it doesn't exist
            # (in the first iteration, or if no submission was blended for a reason) use the mean
            # score.
            previous_blend_action = last_action(ramp_kit_dir, "blend")
            rh.actions.blend(
                ramp_kit_dir = ramp_kit_dir,
                submissions = list(blended_submissions),
                fold_idxs = range(900, 903),
            )
            blend_action = last_action(ramp_kit_dir, "blend")
            if hasattr(blend_action, "blended_score"):
                blended_score = blend_action.blended_score
                contributivities = {
                    s: contributivity_floor + np.array([c for sh, c in blend_action.contributivities.items() if sh[:len(s)] == s]).sum()
                    for s in submissions
                }
                submission_source_f_name = Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
                submission_target_f_name = kaggle_submissions_path / f"auto_{kit_suffix}_{str(round_idx).zfill(3)}.csv"
                kaggle_submit_file(
                    submission_source_f_name = submission_source_f_name,
                    submission_target_f_name = submission_target_f_name,
                    ramp_kit_dir = ramp_kit_dir,
                )
            else:
                blended_score = hyperopt_action.mean_score
                contributivities = {s: 1000 / len(submissions) for s in submissions}
            action_stats[submission].append({
                "runtime": hyperopt_action.runtime,
                "contributivities": contributivities,
            })
        
            # Delete submissions with zero contributivity from the submissions to be blended
            contributivities_df = rh.actions.load_contributivities(ramp_kit_dir)
            non_contributive_submissions = set(contributivities_df[contributivities_df["contributivity"] == 0].index)
            print(f'blended submissions: {blended_submissions}')
            print(f'non-contributive submissions: {non_contributive_submissions}')
            blended_submissions = blended_submissions - non_contributive_submissions
            print(f'new blended submissions: {blended_submissions}')
        #    input("Press Enter to continue...")
    
    for stop_fold_idx in range(904, 900 + n_folds + 1):
        selected_submissions = []
        for submission in submissions:
            rh.actions.select_top_hyperopt_and_train(
                ramp_kit_dir = ramp_kit_dir,
                submission = submission,
                fold_idxs = range(900, stop_fold_idx),
                trained_fold_idxs = range(900, stop_fold_idx - 1),
                top_n = top_n_for_mean,
                n_sigma = n_sigma,
            )
            top_hyperopt_dict = rh.actions.select_top_hyperopt(
                ramp_kit_dir = ramp_kit_dir,
                submission = submission,
                fold_idxs = range(900, stop_fold_idx),
                n_sigma = n_sigma,
            )
            if "selected_submissions" in top_hyperopt_dict:
                selected_submissions += top_hyperopt_dict["selected_submissions"]
        rh.actions.blend(
            ramp_kit_dir = ramp_kit_dir,
            submissions = selected_submissions,
            fold_idxs = range(900, stop_fold_idx),
        )
        submission_source_f_name = Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
        submission_target_f_name = kaggle_submissions_path / f"auto_{kit_suffix}_growing_folds_{str(stop_fold_idx).zfill(2)}.csv"
        kaggle_submit_file(
            submission_source_f_name = submission_source_f_name,
            submission_target_f_name = submission_target_f_name,
            ramp_kit_dir = ramp_kit_dir,
        )
    
    ###############
    
    for stop_fold_idx in range(904, 900 + n_folds + 1):
        for submission in blended_submissions:  # from the last blend of the racing loop
            rh.actions.train(
                ramp_kit_dir = ramp_kit_dir,
                submission = submission,
                fold_idxs = range(900, stop_fold_idx),
            )    
        rh.actions.blend(
            ramp_kit_dir = ramp_kit_dir,
            submissions = list(blended_submissions),
            fold_idxs = range(900, stop_fold_idx),
        )
        submission_source_f_name = Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
        submission_target_f_name = kaggle_submissions_path / f"auto_{kit_suffix}_last_blend_{str(stop_fold_idx).zfill(2)}.csv"
        kaggle_submit_file(
            submission_source_f_name = submission_source_f_name,
            submission_target_f_name = submission_target_f_name,
            ramp_kit_dir = ramp_kit_dir,
        )
    
    # whenever hyperopt submissions are trained outside of hyperopt, we need to update the summary
    for submission in submissions:
        rh.actions.update_hyperopt_score_summary(
            ramp_kit_dir = ramp_kit_dir,
            submission = submission,
        )
        best_submission = rh.actions.select_top_hyperopt(
            ramp_kit_dir = ramp_kit_dir,
            submission = submission,
            fold_idxs = range(900, 900 + n_folds),
            top_n = 1,
        )["selected_submissions"][0]
        submission_source_f_name = Path(ramp_kit_dir) / "submissions" / best_submission / "training_output" / "submission_bagged_test.csv"
        submission_target_f_name = kaggle_submissions_path / f"auto_{kit_suffix}_best_{submission}.csv"
        kaggle_submit_file(
            submission_source_f_name = submission_source_f_name,
            submission_target_f_name = submission_target_f_name,
            ramp_kit_dir = ramp_kit_dir,
        )

def start():
    main()

if __name__ == "__main__":
    start()

