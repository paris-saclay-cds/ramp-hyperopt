import glob
import rampwf as rw
from pathlib import Path
from typing import Tuple, Optional
try:
    from importlib import resources as impresources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as impresources

score_name_type_map = {
    'rmse' : rw.score_types.RMSE,
    'mae' : rw.score_types.MAE,
    'r2' : rw.score_types.R2,
}

def convert_ramp_dirs(ramp_kit_dir: Path | str, ramp_data_dir: Optional[Path | str]) -> Tuple[Path, Path]:
    """Convert ramp dirs to Path.

    Remember that ramp_data_dir does not include the 
    /data subfolder, it is usually the same as ramp_kit_dir,
    but can point to an alternative data source for the same
    kit.
    Args:
        ramp_kit_dir (str): ramp_kit_dir
        ramp_data_dir (str): ramp_data_dir

    Returns:
        (Path, Path): converted dirs
    """
    ramp_kit_dir = Path(ramp_kit_dir)
    if ramp_data_dir is None:
        ramp_data_dir = Path(ramp_kit_dir)
    else:
        ramp_data_dir = Path(ramp_data_dir)
    return ramp_kit_dir, ramp_data_dir


def load_template(package, template_path) -> str:
    """Loads a template from the package

    Args:
        package (_type_): Package providing the template
        template_path (_type_): Template path

    Returns:
        _type_: The code of the template
    """
    templates_dir = impresources.files(package)
    try:
        template_f_name = templates_dir / template_path
        with template_f_name.open("r") as f: 
            template_code = f.read()
    except AttributeError:
        # Python < PY3.9, fall back to method deprecated in PY3.11.
        template_code = impresources.read_text(package / template_path)
    return template_code

def num_data_preprocessors(submission, ramp_kit_dir):
    submission_path = ramp_kit_dir / "submissions" / submission
    dp_idx = 0
    while True:
        submissions_f_names = glob.glob(
            f'{submission_path}/data_preprocessor_{dp_idx}_*.py')
        if len(submissions_f_names) == 0:
            break
        dp_idx += 1
    return dp_idx