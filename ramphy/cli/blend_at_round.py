import click
import glob
import json
import shutil
import numpy as np
import pandas as pd
import rampwf as rw
import ramphy as rh
import ramphy.ramp_setup as rs
from pathlib import Path
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    default=None,
    help="The kit to update.",
)
@click.option(
    "--version",
    default=None,
    help="The program version",
)
@click.option(
    "--number",
    default=None,
    help="The program number (repeated within version)",
)
def main(
    ramp_kit,
    version,
    number,
):
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = f"{ramp_kit}_{kit_suffix}"
    print(ramp_kit_dir)
    metadata = json.load(open(Path(ramp_kit_dir) / "data" / "metadata.json"))
    problem = rw.utils.assert_read_problem(ramp_kit_dir=ramp_kit_dir)
    score_names = [st.name for st in problem.score_types]
    score_type = problem.score_types[-1]
    valid_score_name = f"valid_{score_names[-1]}"
    
    action_f_names = glob.glob(f"{ramp_kit_dir}/actions/*")
    action_f_names.sort()
    ramp_program = []
    for action_f_name in action_f_names:
        f_name = Path(action_f_name).name
        ramp_program.append(rh.actions.load_ramp_action(action_f_name))
    
    hyperopt_actions = [ra for ra in ramp_program if ra.name == "hyperopt"]
    blend_actions = [ra for ra in ramp_program if ra.name == "blend"]
    train_actions = [ra for ra in ramp_program if ra.name == "train"]

    n_folds_hyperopt = 3
    n_rounds = 1
    for blend_action in blend_actions:
        if blend_action.kwargs["fold_idxs"] == range(900, 900 + n_folds_hyperopt):
            n_rounds += 1
            last_race_blend_action = blend_action
    last_race_blended_submissions = [key for key, value in last_race_blend_action.contributivities.items() if value > 0]
    print(last_race_blend_action.__dict__)

    n_folds = 15
    rs.orchestration.train_on_all_folds(
        submissions = last_race_blended_submissions,
        ramp_kit_dir = ramp_kit_dir,
        n_folds = n_folds + 1,
    )
    # Blend then bag the final blend of the hyperopt race on all the folds
    rs.orchestration.final_blend_then_bag(
        submissions = last_race_blended_submissions,
        ramp_kit_dir = ramp_kit_dir,
        kit_suffix = kit_suffix,
        n_folds = n_folds,
        n_rounds = n_rounds,
    )
    # Bag then blend the final blend of the hyperopt race on all the folds
    rs.orchestration.final_bag_then_blend(
        submissions = last_race_blended_submissions,
        ramp_kit_dir = ramp_kit_dir,
        kit_suffix = kit_suffix,
        n_folds = n_folds,
        n_rounds = n_rounds,
    )

def start():
    main()

if __name__ == "__main__":
    start()

