from pathlib import Path
from typing import Dict

from actions import execute_script

import ramphy as rh
from ramphy import ramp_setup as rs


@rh.actions.ramp_action
def pangu_setup_kit(
    pangu_root: str | Path, output_path: str | Path, challenge_name: str, llm: str = "fschat/llama-3-70B-Instruct"
) -> Dict:
    """Runs the pangu auto setup

    Args:
        pangu_root (str | Path): Path to pangu installation
        output_path (str | Path): Path to where to save the setup
        challenge_name (str): Name of the kaggle challenge
        llm (str, optional): LLM config. Defaults to "fschat/llama-3-70B-Instruct".

    Returns:
        Dict: Status of the setup
    """
    env_args = {"HYDRA_FULL_ERROR": 1, "PANGU_DEBUG": 1}
    script_path = Path(pangu_root) / "src" / "pangu" / "start.py"
    hydra_args = {
        "task": "data_preprocessing",
        "llm@agent.llm": llm,
        "method": "data-flow",
        "max_episodes": 1,
        "task.task_url_base": "https://www.kaggle.com/competitions",
        "task.setup_path": str(output_path),
        "task.task_id": challenge_name,
    }

    result = execute_script(script_path=script_path, env_args=env_args, hydra_args=hydra_args, script_args={})
    return {"correctly_executed": result}
