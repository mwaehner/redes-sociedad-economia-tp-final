import os

from run import main

""" Grid Search script to search best NN hyperparameters """

params = {
    "df_dir": "student-por.csv",
    "output_col": "G3",
    "tb_dir": "." + os.sep + "tb_logs",
    "batch_size": 32,
    "seed": 42,
    "use_cpu": False,
    "fast_dev_run": False,
    "cross_validation_folds": 10,
    "num_workers": 1,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "project_name": "portuguese",
    "hidden_size": 10,
    "dropout_rate": 0.0
}

if __name__ == '__main__':
    """Portuguese """
    # Version 0: Hidden size 10, dropout rate None
    main(**params)

    # Version 1: Hidden size 30, dropout rate None
    params["hidden_size"] = 30
    main(**params)

    # Version 2: Hidden size 30, dropout rate 0.5
    params["dropout_rate"] = 0.5
    main(**params)

    """Math """
    params["project_name"] = "math"
    params["df_dir"] = "student-mat.csv"
    params["hidden_size"] = 10
    params["dropout_rate"] = 0.0
    # Version 0: Hidden size 10, dropout rate None
    main(**params)

    # Version 1: Hidden size 30, dropout rate None
    params["hidden_size"] = 30
    main(**params)

    # Version 2: Hidden size 30, dropout rate 0.5
    params["dropout_rate"] = 0.5
    main(**params)


