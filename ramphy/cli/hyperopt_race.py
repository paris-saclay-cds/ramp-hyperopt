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
def kaggle_submit_file(submission_source_f_name, submission_target_f_name, competition, ramp_kit_dir):
    try:
#        kaggle_api = KaggleApi()
#        kaggle_api.authenticate()
        shutil.copy(submission_source_f_name, submission_target_f_name)
#        submission_status = kaggle_api.competition_submit(
#            file_name=submission_target_f_name, message=datetime.datetime.utcnow(),
#            competition=competition)
        return 1
    except:
        return 0

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
def main(
    ramp_kit,
    version,
    number,
):
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = f"{ramp_kit}_{kit_suffix}"

    submissions = ['lgbm', 'xgboost', 'catboost']
    n_rounds = 100
    n_trials_per_round = 15
    top_n_for_mean = 10
    n_sigma = 1
    n_folds = 31
    contributivity_floor = 100  # on 1000, added to contributivity to give a chance to every submission
    first_score = 1000  # should be a bad score
    epsilon = 0  # of eps greedy, to avoid "forgetting" actions due to early failures
    criterion = "contributivity"
    
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    score_names = [st.name for st in problem.score_types] 
    valid_score_name = f'valid_{score_names[0]}'
    is_lower_the_better = problem.score_types[0].is_lower_the_better
    
    for submission in submissions:
        rh.ramp_setup.setup_scripts.tabular_regression.tabular_regression_columnwise_first_submit(         
            ramp_kit_dir = ramp_kit_dir,
            submission = submission,
            regressor = submission,
        )                                                   
    
    
    with open(Path(ramp_kit_dir) / "data" / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    kaggle_n_submissions_per_day = 100
    if "kaggle_n_submissions_per_day" in metadata.keys():
        kaggle_n_submissions_per_day = metadata["kaggle_n_submissions_per_day"]
    kaggle_n_submissions = 0
    kaggle_submissions_path = Path(ramp_kit_dir) / "kaggle_submissions"
    kaggle_submissions_path.mkdir(parents=False, exist_ok=True)
#    kaggle_api = KaggleApi()
#    kaggle_api.authenticate()
#    
#    try:
#        private_leaderboard_scores = rh.ramp_setup.actions.get_private_leaderboard_scores(
#            kaggle_api=kaggle_api, competition=metadata["kaggle_name"])
#        np.save(Path(ramp_kit_dir) / "data" / "private_leaderboard_scores.npy", private_leaderboard_scores)
#    except:
#        pass
#    try:
#        public_leaderboard_scores = rh.ramp_setup.actions.get_public_leaderboard_scores(
#            kaggle_api=kaggle_api, competition=metadata["kaggle_name"])
#        np.save(Path(ramp_kit_dir) / "data" / "public_leaderboard_scores.npy", public_leaderboard_scores)
#    except:
#        pass
    
    #Dictionary of submissions: list of dictionary of run times and scores
    action_stats = {submission: [] for submission in submissions}
    blended_submissions = set()
    
    improvement_speed_df = pd.DataFrame(columns=["round"] + submissions)
    for round_idx in range(n_rounds):
        # We'll choose the submission that improves the results the fastest historically
        # Different ways to define improvement
        improvement_speeds = {}
        for submission in submissions:
            ast = action_stats[submission]
            if criterion == "last step":
                # the improvement of the last step, floored at zero
                if len(ast) >= 2:
                    if is_lower_the_better:
                        best_score = min([a["mean_score"] for a in ast[:-1]])
                        speed = (best_score - ast[-1]["mean_score"]) / ast[-1]["runtime"].total_seconds()
                    else:
                        best_score = max([a["mean_score"] for a in ast[:-1]])
                        speed = (ast[-1]["mean_score"] - best) / ast[-1]["runtime"].total_seconds()
                    speed = max(0, speed)
                else:
                    speed = np.finfo(float).max
            elif criterion == "since the beginning - mean score":
                # improvement of the action (submission) since the beginning
                if len(ast) >= 1:
                    total_time = np.array([a["runtime"].total_seconds() for a in ast]).sum()
                    if is_lower_the_better:
                        best_score = min([a["mean_score"] for a in ast])
                        speed = (first_score - best_score) / total_time
                    else:
                        best_score = max([a["mean_score"] for a in ast])
                        speed = (best_score - first_score) / total_time
                else:
                    speed = np.finfo(float).max            
            elif criterion == "since the beginning - blended score improvement":
                # sum of improvements of the blended score, applying this action
                if len(ast) >= 2:
                    total_time = np.array([a["runtime"].total_seconds() for a in ast]).sum()
                    total_score = np.array([a["blended_score_improvement"] for a in ast]).sum()
                    speed = total_score / total_time
                    if is_lower_the_better:
                        speed *= -1
                else:
                    speed = np.finfo(float).max            
            elif criterion == "contributivity":
                # sum of improvements of the blended score, applying this action
                if len(ast) >= 2:
                    last_ast = ast[-1]
                    total_time = np.array([a["runtime"].total_seconds() for a in ast]).sum()
                    contributivity = last_ast["contributivities"][submission]
                    speed = contributivity / total_time
                else:
                    speed = np.finfo(float).max            
            improvement_speeds[submission] = speed
        print(f'action_stats:\n{action_stats}')
        print(f'improvement_speeds:\n{improvement_speeds}')
        row = pd.DataFrame(dict({"round_idx" : round_idx}, **improvement_speeds), index=[round_idx])
        improvement_speed_df = pd.concat([improvement_speed_df, row], ignore_index=True)
        improvement_speed_df.to_csv("improvement_speeds.csv")
        max_speed = max(improvement_speeds.values())
        best_submissions = [submission for submission, speed in improvement_speeds.items() if speed == max_speed]
        if epsilon > random.random():
            print("Eps-greedy choice")
            best_submissions = submissions
        submission = random.choice(best_submissions)  # in case of tie (at zero typically), or eps greedy, random choice
        print(f'best_submissions: {best_submissions}')
        print(f'selected submission : {submission}')
    #    input("Press Enter to continue...")
        if round_idx == 0:
            n_trials = 15
        else:
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
            if not previous_blend_action is None and hasattr(previous_blend_action, "blended_score"):
                previous_blended_score = previous_blend_action.blended_score
            else:
                previous_blended_score = hyperopt_action.mean_score
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
                # Submitting kaggle_n_submissions_per_day - 2 times, letting 2 submissions for the final scores
                if blended_score != previous_blended_score: # and kaggle_n_submissions * n_rounds / (kaggle_n_submissions_per_day - 2) <= round_idx:
                    kaggle_n_submissions += 1
                    submission_source_f_name = Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
                    submission_target_f_name = kaggle_submissions_path / f"auto_{kit_suffix}_{str(round_idx).zfill(3)}.csv"
                    kaggle_submit_file(
                        submission_source_f_name = submission_source_f_name,
                        submission_target_f_name = submission_target_f_name,
                        competition = metadata["kaggle_name"],
                        ramp_kit_dir = ramp_kit_dir,
                    )
            else:
                blended_score = hyperopt_action.mean_score
                contributivities = {s: 1000 / len(submissions) for s in submissions}
            blended_score_improvement = blended_score - previous_blended_score
            action_stats[submission].append({
                "best_score" : hyperopt_action.mean_score,
                "blended_score_improvement" : blended_score_improvement,  # can be "negative" if blending doesn't improve
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
    
    #for submission in submissions:
    #    rh.actions.delete_duplicates_hyperopt(
    #        ramp_kit_dir = ramp_kit_dir,
    #        submission = submission,
    #        fold_idxs = range(900, 903),
    #    )
    
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
            competition = metadata["kaggle_name"],
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
            competition = metadata["kaggle_name"],
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
            competition = metadata["kaggle_name"],
            ramp_kit_dir = ramp_kit_dir,
        )

def start():
    main()

if __name__ == "__main__":
    start()

