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
    'rmsle': rw.score_types.RMSLE,
}

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