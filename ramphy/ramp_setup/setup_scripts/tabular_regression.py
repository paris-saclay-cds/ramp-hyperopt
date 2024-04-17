import json
import numpy as np
import pandas as pd
from pathlib import Path
import rampwf as rw
import ramphy as rh

def tabular_regression_setup(
    ramp_setup_kit_dir,
    ramp_kit_dir = '.',
    ramp_data_dir = None,
    ramp_templates_dir = '/nas/ramp-hyperopt/ramphy/ramp_setup/',
) -> None:
    problem_template_f_name = Path(ramp_templates_dir) / 'problems' / 'tabular_regression_problem.py'
    
    ramp_setup_kit_dir = Path(ramp_setup_kit_dir)
    ramp_kit_dir = Path(ramp_kit_dir)
    if ramp_data_dir is None:
        ramp_data_dir = Path(ramp_kit_dir)
    else:
        ramp_data_dir = Path(ramp_data_dir)
        
    ramp_kit_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_f_name = ramp_setup_kit_dir / 'metadata.json'
    problem_f_name = ramp_kit_dir / 'problem.py'
    
    problem_code = open(problem_template_f_name).read()
    metadata = json.load(open(metadata_f_name))
    problem_code = problem_code.format(**metadata)
    with open(problem_f_name, 'w') as f_out:
        f_out.write(problem_code)
    (ramp_data_dir / 'data').mkdir(parents=True, exist_ok=True)
    (ramp_kit_dir / 'submissions').mkdir(parents=True, exist_ok=True)
    
    train_data = pd.read_csv(ramp_setup_kit_dir / 'train.csv')
    test_data = pd.read_csv(ramp_setup_kit_dir / 'test.csv')
    sample_submission = pd.read_csv(ramp_setup_kit_dir / 'sample_submission.csv')
    
    metadata['col_values'] = {}
    for col, col_type in metadata['col_types'].items():
        if col_type == 'cat' or col_type == 'bin':
            # Ensure the column is treated as string to safely use .str accessor
            train_data[col] = train_data[col].astype(str)
            train_data[col] = train_data[col].str.strip()
            test_data[col] = test_data[col].astype(str)
            test_data[col] = test_data[col].str.strip()
            metadata['col_values'][col] = sorted(pd.concat([train_data[col], test_data[col]]).dropna().unique().tolist())
    
    # mock test labels
    # matching mean and sigma from training set
    np.random.seed(43)
    test_data[metadata['target_col']] = np.random.normal(
        train_data[metadata['target_col']].mean(), train_data[metadata['target_col']].std())
    test_data.to_csv(ramp_data_dir / 'data' / 'test.csv', index=False)
    train_data.to_csv(ramp_data_dir / 'data' / 'train.csv', index=False)
    sample_submission.to_csv(ramp_data_dir / 'data' / 'sample_submission.csv', index=False)
    if metadata['score_name'] in ['mse', 'rmse']: 
        metadata['lgbm_objective'] = 'mse'            
    elif metadata['score_name'] == 'mae':         
        metadata['lgbm_objective'] = 'mae'            
    
    with open(ramp_data_dir / 'data' / 'metadata.json', 'w') as f_out:
        json.dump(metadata, f_out)
    
def tabular_regression_submit(
    submission,
    workflow_element_dict = {
        'regressor': 'lgbm',
        'feature_extractor': 'empty',
        'data_preprocessors': ['drop_id', 'invalid_col_names', 'cat_col_encoding',]
    },
    ramp_kit_dir = '.',
    ramp_data_dir = None,
    ramp_templates_dir = '/nas/ramp-hyperopt/ramphy/ramp_setup/',
) -> None:
    ramp_templates_dir = Path(ramp_templates_dir)
    ramp_kit_dir = Path(ramp_kit_dir)
    if ramp_data_dir is None:
        ramp_data_dir = Path(ramp_kit_dir)
    else:
        ramp_data_dir = Path(ramp_data_dir)

    (ramp_kit_dir / 'submissions' / submission).mkdir(parents=True, exist_ok=True)
    metadata_f_name = ramp_data_dir / 'data' / 'metadata.json'
    metadata = json.load(open(metadata_f_name))
    
    regressor_f_name = ramp_templates_dir / 'workflow_elements' / 'tabular_regressors' / f'{workflow_element_dict["regressor"]}.py'
    regressor_code = open(regressor_f_name).read()
    regressor_code = regressor_code.format(**metadata)
    with open(ramp_kit_dir / 'submissions' / submission / 'regressor.py', 'w') as f_out:
        f_out.write(regressor_code)

    fe_f_name = ramp_templates_dir / 'workflow_elements' / 'tabular_feature_extractors' / f'{workflow_element_dict["feature_extractor"]}.py'
    fe_code = open(fe_f_name).read()
    fe_code = fe_code.format(**metadata)
    with open(ramp_kit_dir / 'submissions' / submission / 'feature_extractor.py', 'w') as f_out:
        f_out.write(fe_code)
    
    for i, dp in enumerate(workflow_element_dict['data_preprocessors']):
        dp_f_name = ramp_templates_dir / 'workflow_elements' / 'tabular_data_preprocessors' / f'{dp}.py'
        dp_code = open(dp_f_name).read()
        dp_code = dp_code.format(**metadata)
        with open(ramp_kit_dir / 'submissions' / submission / f'data_preprocessor_{i}_{dp}.py', 'w') as f_out:
            f_out.write(dp_code)
