import os
import glob
import pytest
import numpy as np
import ramphy as rh
from pathlib import Path

PATH = os.path.dirname(__file__)


def _generate_grid_path_kits():
    grid = []
    for path_kit in sorted(glob.glob(os.path.join(PATH, 'ramp_setup_kits', '*'))):
        grid.append(os.path.abspath(path_kit))
    return grid


@pytest.mark.parametrize(
    "path_kit",
    _generate_grid_path_kits()
)
def test_submission(path_kit):
    ramp_kit_dir = Path(PATH) / 'ramp_kits' / 'test_kit'
    ramp_kit_dir.mkdir(parents=True, exist_ok=True)
    rh.kit_setup(
        download_dir = path_kit,
        ramp_kit_dir = ramp_kit_dir,
        ramp_data_dir = ramp_kit_dir,
        ramp_templates_dir = None,
    )
    rh.actions.train(
        submission = 'starting_kit',
        fold_idxs = [400, 401],
        force_retrain = True,
        ramp_kit_dir = ramp_kit_dir,
        ramp_data_dir = ramp_kit_dir,
    )

