RAMP hyperopt
=============

Hyperopt package for ramp-workflow.

# Command-line parameters

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

