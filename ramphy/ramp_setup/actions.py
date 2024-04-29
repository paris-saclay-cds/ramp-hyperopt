from pathlib import Path
import json
from typing import Optional, Callable
from ramphy import ramp_setup as rs

def submit_llm_feature_rejector(asking_function: Callable, 
                                parsing_function: Callable,
                                prompting_function: Callable, 
                                ramp_kit_dir: Path | str, 
                                submission:  Path | str,
                                ramp_data_dir: Optional[str | Path] = None):
    """This function asks the llm for a feature to drop and submits a preprocessor that does that

    Args:
        asking_function (Callable): Function used to ask the LLM
        parsing_function (Callable): Function used to parse the LLM answer. It has to return the name of the feature to drop
        prompting_function (Callable): Function to generate the prompt from the prompt template
        ramp_kit_dir (Path | str): Path of the ramp kit dir
        submission (Path | str): Name of the submission
        ramp_data_dir (Optional[str  |  Path], optional): Path of the ramp data dir. Defaults to None.
    """
    ramp_kit_dir, ramp_data_dir = rs.utils.convert_ramp_dirs(ramp_kit_dir, ramp_data_dir)
    (ramp_kit_dir / "submissions" / submission).mkdir(parents=True, exist_ok=True)
    metadata = json.load(open(ramp_data_dir / "data" / "metadata.json"))

    # Ask the LLM which feature to drop
    # TODO prepare prompt
    prompt = prompting_function(['prompts/drop_feature.jinja'], {'metadata': metadata})
    feature_to_drop = parsing_function(asking_function(prompt))

    dp_idx = rs.utils.num_data_preprocessors(submission, ramp_kit_dir)

    # Load and prepare the preprocessor from the template
    dp_template_path = Path("workflow_elements") / "tabular_data_preprocessors" / "llm_drop_column.py"
    dp_code = rs.utils.load_template(package=rs, template_path=dp_template_path)
    dp_code = dp_code.format(dropped_feat=feature_to_drop)

    # Add the preprocessor in the submission folder
    with open(ramp_kit_dir / "submissions" / submission / f"data_preprocessor_{dp_idx}_llm_drop_{feature_to_drop}.py", "w") as f_out:
        f_out.write(dp_code)


