import abc
import os
from copy import deepcopy
from os import listdir
from os.path import isdir, isfile, join

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks import ClassificationMetricsCallback, ModelWeightsHistogramCallback
from k_fold_data_module import KFoldDataModule
from students_dataset import StudentsDataset
from model import Model


class ExperimentRunner:
    """ Base class that encapsulates the shared logic to run experiments. Each experiment should have a specific
    subclass and implement all the abstract methods. """

    def __init__(self, df_dir, output_col, tb_dir, num_epochs, batch_size, seed, use_cpu,
                 fast_dev_run, num_workers, cross_validation_folds, project_name, learning_rate, hidden_size,
                 dropout_rate):
        self.df_dir = df_dir
        self.output_col = output_col
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.use_cpu = use_cpu
        self.fast_dev_run = fast_dev_run  # Use this param if you want to test with a fast but incomplete run
        self.cross_validation_folds = cross_validation_folds
        self.num_workers = num_workers  # Number of producer workers in the dataset
        self.project_logs_dir = os.path.join(tb_dir, project_name)

        self.logger_args = {
            "save_dir": tb_dir,
            "name": project_name,
            "log_graph": True,
            "default_hp_metric": False
        }

        self.dataset = self.get_dataset(self.df_dir, output_col)
        self.model = self.get_model(learning_rate, hidden_size, dropout_rate)
        self.datamodule = self.get_datamodule(self.dataset)
        self.starting_fold = 0
        self.trainer_args = self.get_trainer_args()

    def run(self):
        """ Trains and validates the model using a cross validation loop. """
        torch.cuda.empty_cache()

        # Used to log architecture graph in Tensorboard
        self.model.example_input_array = self.datamodule.get_example_input()

        callbacks = self.get_callbacks()

        lightning_module_state_dict = deepcopy(self.model.state_dict())

        for fold_no in range(self.starting_fold, self.cross_validation_folds):
            print("STARTING FOLD ", fold_no)
            self.datamodule.setup_fold_index(fold_no)
            self.model.load_state_dict(lightning_module_state_dict)  # Resets model weights
            self.model.set_fold(fold_no)
            trainer = Trainer(**self.trainer_args, callbacks=callbacks)
            trainer_fit_args = {
                "model": self.model,
                "datamodule": self.datamodule,
            }
            trainer.fit(**trainer_fit_args)

    def get_datamodule(self, dataset):
        datamodule = KFoldDataModule(
            dataset,
            batch_size=self.batch_size,
            seed=self.seed,
            num_workers=self.num_workers,
            number_of_folds=self.cross_validation_folds
        )
        return datamodule

    def get_trainer_args(self):
        logger = TensorBoardLogger(
            **self.logger_args
        )
        trainer_args = {
            "max_epochs": self.num_epochs,
            "logger": logger,
            "log_every_n_steps": 2,
            "num_sanity_val_steps": 0
        }

        trainer_args["devices"] = 1

        if self.use_cpu:
            trainer_args["accelerator"] = "cpu"
        else:
            trainer_args["accelerator"] = "auto"

        if self.fast_dev_run:
            trainer_args["limit_train_batches"] = 2
            trainer_args["limit_val_batches"] = 2
            trainer_args["limit_test_batches"] = 2
        return trainer_args

    def get_dataset(self, data_dir, output_col):
        return StudentsDataset(
            data_dir,
            output_col
        )

    def get_model(self, learning_rate, hidden_size, dropout_rate):
        return Model(self.use_cpu, learning_rate, hidden_size=hidden_size, dropout_rate=dropout_rate)

    def get_callbacks(self):
        callbacks = [
            ClassificationMetricsCallback(self.cross_validation_folds),
            ModelWeightsHistogramCallback(),
        ]
        return callbacks