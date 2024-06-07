import json
import click
from pathlib import Path
import ramphy as rh
rh.actions.EXECUTE_PLAN = True
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


# flake8: noqa: E501


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    help="The kit to set up.",
)
@click.option(
    "--setup-dir",
    default="/nas/ramp-setup-kits",
    show_default=True,
    help="The root folder where the kits with original metadata and "
    "train/test files are",
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
    setup_dir,
    version,
    number,
):
    kit_suffix = f"v{version}_n{number}"
    ramp_kit_dir = f"{ramp_kit}_{kit_suffix}"
    
    rh.ramp_setup.setup_scripts.tabular.setup(
        download_dir = f"{setup_dir}/{ramp_kit}",
        ramp_kit_dir = ramp_kit_dir,
    )

    metadata = json.load(open(Path(ramp_kit_dir) / "data" / "metadata.json"))

    if "regression" in metadata["prediction_type"]:
        rh.ramp_setup.setup_scripts.tabular.tabular_regression_columnwise_last_submit(         
            ramp_kit_dir = ramp_kit_dir,
            submission = 'starting_kit',
            regressor = 'lgbm',
        )   
    elif "classification" in metadata["prediction_type"]:
        rh.ramp_setup.setup_scripts.tabular.tabular_classification_columnwise_last_submit(         
            ramp_kit_dir = ramp_kit_dir,
            submission = 'starting_kit',
            classifier = 'lgbm',
        )
                                                                                                       
    rh.actions.train(                                                                                  
        ramp_kit_dir = ramp_kit_dir,    
        submission = 'starting_kit',                                                                       
        fold_idxs = range(900, 903),                                                                   
        force_retrain = True,
    )                                                                                                  

def start():
    main()

if __name__ == "__main__":
    start()

