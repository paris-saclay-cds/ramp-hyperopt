import json
from pathlib import Path
from typing import Callable, Dict, Optional

from kaggle.api.kaggle_api_extended import KaggleApi

import ramphy as rh
from ramphy import ramp_setup as rs


@rh.actions.ramp_action
def submit_llm_feature_rejector(
    asking_function: Callable,
    parser_function: Callable,
    pronpt_builder: Callable,
    ramp_kit_dir: Path | str,
    submission: Path | str,
    ramp_data_dir: Optional[str | Path] = None,
):
    """This function asks the llm for a feature to drop and submits a preprocessor that does that

    Args:
        asking_function (Callable): Function that uses the LLM to generate and answer
        parser_function (Callable): Function used to parse the LLM answer. It has to return the name of the feat to drop
        pronpt_builder (Callable): Function to generate the prompt from the prompt template
        ramp_kit_dir (Path | str): Path of the ramp kit dir
        submission (Path | str): Name of the submission
        ramp_data_dir (Optional[str  |  Path], optional): Path of the ramp data dir. Defaults to None.
    """
    ramp_kit_dir, ramp_data_dir = rh.actions.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    # Ask the LLM which feature to drop
    # TODO prepare prompt
    # TODO need task description and challenge description in the metadata
    # Also need a view of the dataset (maybe df.head)
    prompt = pronpt_builder(["prompts/drop_feature.jinja"], {"metadata": metadata})
    feature_to_drop = parser_function(asking_function(prompt))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)

    # Load and prepare the preprocessor from the template
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "llm_drop_column.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    dp_code = dp_code.format(dropped_feat=feature_to_drop)

    # Add the preprocessor in the submission folder
    with open(
        ramp_kit_dir / "submissions" / submission / f"data_preprocessor_{dp_idx}_llm_drop_{feature_to_drop}.py", "w"
    ) as f_out:
        f_out.write(dp_code)


@rh.actions.ramp_action
def kaggle_submit(
    submission: str,
    ramp_kit_dir: Path | str,
    kaggle_name: str,
    submission_description: Optional[str] = None,
) -> Dict:
    """Submits the predictions to Kaggle

    Args:
        submission (str): RAMP Submission name. If blended it submits the blended results
        ramp_kit_dir (Path | str): Path to ramp kit
        competition_name (str): Kaggle competition name
        submission_description (Optional[str], optional): Description to Kaggle submission. Defaults to None.

    Returns:
        Dict: _description_
    """
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    action_output = {}

    if submission == "blended":
        file_path = Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
        assert file_path.exists(), "No blended test data found."
        submission_description = f"Blended {Path(ramp_kit_dir).name}"
        submission_status = kaggle_api.competition_submit(
            file_name=file_path, message=submission_description, competition=kaggle_name
        )
        action_output["submission_status"] = submission_status
        action_output["kaggle_submission"] = submission_description
    else:
        if submission_description is None:
            submission_description = submission

        file_path = Path(ramp_kit_dir) / "submissions" / submission / "training_output" / "submission_bagged_test.csv"
        assert Path(ramp_kit_dir) / "submissions" / submission, f"Submission {submission} does not exists."
        assert file_path.exists(), f"File {file_path} does not exists. Sure that the submission has been trained?"
        submission_status = kaggle_api.competition_submit(
            file_name=file_path, message=submission_description, competition=kaggle_name
        )
        action_output["submission_status"] = submission_status
        action_output["kaggle_submission"] = submission_description

    leaderboard_raw = kaggle_api.competition_leaderboard_view(competition=kaggle_name)
    leaderboard = {sub.teamName: sub.score for sub in leaderboard_raw}
    action_output["leaderboard"] = leaderboard

    submissions_raw = kaggle_api.competition_submissions(competition=kaggle_name)
    action_output["submissions"] = submissions_raw

    return action_output
