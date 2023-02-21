import argparse
import os

import torch

from experiment_runner import ExperimentRunner


def main(df_dir, output_col, tb_dir, num_epochs, batch_size, seed, use_cpu,
         fast_dev_run,
         cross_validation_folds, num_workers, learning_rate, project_name, hidden_size, dropout_rate):
    torch.cuda.empty_cache()

    # More experiments_runners to be added here
    experiment_args = {
        "df_dir": df_dir,
        "output_col": output_col,
        "tb_dir": tb_dir,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "use_cpu": use_cpu,
        "fast_dev_run": fast_dev_run,
        "cross_validation_folds": cross_validation_folds,
        "num_workers": num_workers,
        "learning_rate": learning_rate,
        "project_name": project_name,
        "hidden_size": hidden_size,
        "dropout_rate": dropout_rate
    }

    experiment = ExperimentRunner(**experiment_args)
    experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train portuguese students model.')
    parser.add_argument(
        '--df_dir',
        type=str,
        default="student-por.csv",
        help='Dataframe directory'
    )
    parser.add_argument(
        '--output_col',
        type=str,
        default="G3",
        help='The path to the directory containing the data to train the model'
    )
    parser.add_argument(
        '--tb_dir',
        type=str,
        default="." + os.sep + "tb_logs",  # "./tb_logs" for Linux & MacOS, ".\tb_logs" for Windows
        help='The path to the directory where to store all the tensorboard logs'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Seed to split the dataset"
    )
    parser.add_argument(
        '--use_cpu',
        action='store_true',
        default=False,
        help="To run project with CPU, otherwise, uses GPU if available"
    )
    parser.add_argument(
        '--fast_dev_run',
        type=bool,
        default=False,
        help="Turn on for a fast test run"
    )
    parser.add_argument(
        '--cross_validation_folds',
        type=int,
        default=10,
        help="Number of cross validation folds"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help="Number of producer workers in the dataset"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        '--project_name',
        type=str,
        default="portuguese",
        help="Project name"
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=30,
        help="Hidden layer size"
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.5,
        help="Dropout rate"
    )

    args = parser.parse_args()

    main(**vars(args))
