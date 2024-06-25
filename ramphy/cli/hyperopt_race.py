import ramphy.ramp_setup as rs
import click
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
def main(
    ramp_kit,
    kit_root,
    version,
    number,
    resume,
    n_rounds,
):
    rs.orchestration.hyperopt_race(
        ramp_kit = ramp_kit,
        kit_root = kit_root,
        version = version,
        number = number,
        resume = resume,
        n_rounds = n_rounds,
    )

def start():
    main()

if __name__ == "__main__":
    start()

