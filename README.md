RAMP hyperopt
=============

Hyperopt package for ramp-workflow.

# Installation

# How to use

You need a RAMP kit set up, for example, from [here](https://github.com/ramp-kits/kaggle_abalone), with a submission to be hyperopted. Each workflow element can have a section describing the hyperparameters and grids of values, see for example the [regressor](https://github.com/ramp-kits/kaggle_abalone/blob/master/submissions/starting_kit/regressor.py). You can hyperopt each workflow element separately, or at the same time. You can start by 
```
ramp-hyperopt --config config_hyperopt_1.ini
```
using the config file [here](https://github.com/ramp-kits/kaggle_abalone/blob/master/config_hyperopt_1.ini). It will run three trials on the first three folds from the CV object defined in [`problem.py`](https://github.com/ramp-kits/kaggle_abalone/blob/master/problem.py#L30), using the [HEBO](https://github.com/huawei-noah/HEBO/tree/master/HEBO) hyperopt engine. 

## Command-line parameters

`--engine`:
- `ray_hebo` (suggested)
- `ray_zoopt`
- `ray_optuna` (does not work with `--resume`)
- `ray_ax` (does not work: AssertionError: Experiment not set on Ax client)
- `ray_skopt` (does not work: does not accept integer grids with a single element, used to restrict sampling in partial hyperopt)
- `ray_hyperopt` (does not work: does not accept integer grids with a single element, used to restrict sampling in partial hyperopt)
- `ray_bayesopt` (does not work: AttributeError: module 'bayes_opt' has no attribute 'UtilityFunction')
- `ray_bohb` (does not work, can't install: ERROR: Could not build wheels for netifaces)
- `ray_nevergrad` (does not work: does not accept integer grids with a single element, used to restrict sampling in partial hyperopt)

