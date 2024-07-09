import json
import numpy as np
import ramphy as rh
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
rh.actions.EXECUTE_PLAN = True
import click
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


# flake8: noqa: E501

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    help="The kit to download.",
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
    
    with open(Path(ramp_kit_dir) / "data" / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    
    try:
        private_leaderboard_scores = rh.ramp_setup.actions.get_leaderboard_scores(
            kaggle_api=kaggle_api, competition=metadata["kaggle_name"], phase="private")
        np.save(Path(ramp_kit_dir) / "data" / "private_leaderboard_scores.npy", private_leaderboard_scores)
    except Exception as e:
        print(e)
    try:
        public_leaderboard_scores = rh.ramp_setup.actions.get_leaderboard_scores(
            kaggle_api=kaggle_api, competition=metadata["kaggle_name"], phase="public")
        np.save(Path(ramp_kit_dir) / "data" / "public_leaderboard_scores.npy", public_leaderboard_scores)
    except Exception as e:
        print(e)

def start():
    main()

if __name__ == "__main__":
    start()

