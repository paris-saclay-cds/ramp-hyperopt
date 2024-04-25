import json
from pathlib import Path
from typing import Optional
import ramphy.ramp_setup as rs


def kit_setup(
    download_dir: str | Path,
    ramp_kit_dir: str | Path = ".",
    ramp_data_dir: Optional[str | Path] = None,
    ramp_templates_dir: Optional[str | Path] = None,
) -> None:
    """This function calls the challenge-type specific setup function

    Args:
        ramp_setup_kit_dir (str | Path): Path of where the data and metadata have been downloaded
        ramp_kit_dir (str | Path, optional): Path of where to make the kit dir. Defaults to ".".
        ramp_data_dir (Optional[str  |  Path], optional): Path of where to save the data. If None it's ramp_kit_dir/data. Defaults to None.
        ramp_templates_dir (str | Path, optional): Path of where the templates are. Defaults to None. If None it uses the package dir
    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    download_dir = Path(download_dir)
    ramp_kit_dir = Path(ramp_kit_dir)
    if ramp_data_dir is not None:
        ramp_data_dir = Path(ramp_data_dir)
    if ramp_templates_dir is None:
        ramp_templates_dir = Path(__file__).parent.parent
    else:
        ramp_templates_dir = Path(ramp_templates_dir)

    # Maybe the challenge type can be passed from outside...
    #    metadata = load_metadata_from_json(download_dir)
    metadata = json.load(open(download_dir / "metadata.json"))
    prediction_type = metadata["prediction_type"]
    if metadata["score_name"] in ["mse", "rmse", "mae"]:
        prediction_type = "regression"
    feature_types = list(metadata["data_description"]["feature_types"].values())
    input_types = set()
    if "num" in feature_types:
        input_types.add("tabular")
    if "cat" in feature_types:
        input_types.add("tabular")

    if prediction_type == "regression" and len(input_types) == 1 and "tabular" in input_types:
        # Setup the challenge kit
        rs.tabular_regression_setup(
            download_dir=download_dir,
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
            ramp_templates_dir=ramp_templates_dir,
        )
        # Add the starting kit
        rs.tabular_regression_basic_submit(
            submission="starting_kit",
            ramp_kit_dir=ramp_kit_dir,
            ramp_data_dir=ramp_data_dir,
            ramp_templates_dir=ramp_templates_dir,
        )
    else:
        raise NotImplementedError("For now only regression setup is implemented!")
