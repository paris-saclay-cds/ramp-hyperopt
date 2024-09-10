import ramphy.ramp_setup as rs
import click
import click_config_file

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# flake8: noqa: E501


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--ramp-kit",
    help="The kit to hyperopt.",
)
@click.option(
    "--kit-root",
    default="./",
    show_default=True,
    help="The root folder where the kits are",
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
@click.option(
    "--n-rounds",
    default=100,
    show_default=True,
    help="The number of hyperopt rounds",
)
@click.option(
    "--n-trials-per-round",
    default=5,
    show_default=True,
    help="The number of trials per hyperopt round",
)
@click.option(
    "--patience",
    default=-1,
    show_default=True,
    help="The number of rounds after wwhich we stop if score does not improve.",
)
@click.option(
    "--no-growing-folds",
    is_flag=True,
    default=True,
    show_default=True,
    help="Do not run the growing fold submission at the end.",
)
@click.option(
    "--base_predictors",
    multiple=True,
    default=["lgbm", "xgboost", "catboost"],
    help="A list of base predictors.",
)
@click.option(
    "--data_preprocessors",
    multiple=True,
    default=["drop_id", "drop_columns", "base_columnwise"],
    help="A list of data_preprocessors to use. base_columnwise are the base col encoding and inputing",
)
@click.option(
    "--preprocessors_to_hyperopt",
    multiple=True,
    default=None,
    help="A list of preprocessors to hyperopt.",
)
@click_config_file.configuration_option()
def main(
    ramp_kit,
    kit_root,
    version,
    number,
    resume,
    n_rounds,
    n_trials_per_round,
    patience,
    no_growing_folds,
    base_predictors,
    data_preprocessors,
    preprocessors_to_hyperopt,
):
    rs.orchestration.hyperopt_race(
        data_preprocessors=list(data_preprocessors),
        ramp_kit=ramp_kit,
        kit_root=kit_root,
        version=version,
        number=number,
        resume=resume,
        n_rounds=n_rounds,
        n_trials_per_round=n_trials_per_round,
        patience=patience,
        no_growing_folds=no_growing_folds,
        preprocessors_to_hyperopt=list(preprocessors_to_hyperopt),
        base_predictors=list(base_predictors),
    )


def start():
    main()


if __name__ == "__main__":
    start()
