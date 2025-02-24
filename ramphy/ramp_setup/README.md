RAMP AutoDS
=============

Automated Tabular Data Scientist based on the RAMP ecosystem.

# Installation

# Setup

1. Clone a RAMP setup kit, for example, [`kaggle_abalone`](https://github.com/ramp-setup-kits/kaggle_abalone), place it in `./ramp-setup-kits` (by convention), and download the data from the kaggle site.
```
ramp-setup-kits
└───kaggle_abalone
    ├───metadata.json
    ├───sample_submission.csv
    ├───test.csv
    └───train.csv
```

2. If you haven't, create the `ramp-kits` folder
```
mkdir ramp-kits
```

3. Set up the first run on your new kit:
```
cd ramp-kits
ramp-setup --ramp-kit kaggle_abalone --version 1_1 --number 1
```
`version` and `number` can be any string. Conventionally we use `version` to mark either a version of `ramp-autods` or a config file specifying command-line parameters, and `number` to mark an execution, like a seed.

If your setup folder is not `./ramp-setup-kits`, you can specify it with `--setup-root`.

The result is a functional RAMP kit, with the starting kit submission (an LGBM) trained, tested, and scored.
```
ramp-kits
└───kaggle_abalone_v1_1_n1
    ├───actions
    │   ├───'2025-02-24 16:57:58.900437.pkl'
    │   └───...
    ├───data
    │   ├───metadata.json
    │   ├───sample_submission.csv
    │   ├───test.csv
    │   └───train.csv
    ├───problem.py
    └───submissions
        └───starting_kit
            ├───data_preprocessor_0_drop_id.py
            ├───data_preprocessor_1_drop_columns.py
            ├───data_preprocessor_2_col_in_train_only.py
            ├───data_preprocessor_3_Sex_1_cat_col_encoding.py
            ├───data_preprocessor_4_rm_constant_col.py
            ├───feature_extractor.py
            ├───regressor.py
            └───training_output
                └───starting_kit
                    ├───bagged_scores.csv
                    ├───fold_0
                    │   ├───scores.csv
                    │   ├───y_pred_test.npz
                    │   └───y_pred_train.npz
                    ├───...
                    ├───submission_bagged_test.csv 
                    └───submission_bagged_valid.csv
      
```
