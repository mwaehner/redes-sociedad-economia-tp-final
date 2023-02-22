from typing import Optional

import tensorboard_reducer as tbr
from pathlib import Path
import pytorch_lightning as pl
import torch
import numpy as np
from torchmetrics import Accuracy, Recall, Precision
from torchmetrics.functional import f1_score
from torchmetrics.functional import auroc
from sklearn.metrics import RocCurveDisplay


class ModelWeightsHistogramCallback(pl.Callback):
    def on_train_epoch_end(self, pl_trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for name, params in pl_module.named_parameters():
            pl_trainer.logger.experiment.add_histogram(name, params, pl_module.current_epoch)


class ClassificationMetricsCallback(pl.Callback):
    def __init__(self, fold_numbers):
        super(ClassificationMetricsCallback, self).__init__()
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_precision = Precision(task="binary")
        self.fold_numbers = fold_numbers

        if torch.cuda.is_available():
            self.train_acc.cuda()
            self.val_acc.cuda()
            self.val_recall.cuda()
            self.val_precision.cuda()

    def on_validation_epoch_end(self, pl_trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        y = pl_module.all_validation_ys.int()
        y_hat = pl_module.all_validation_y_hats
        metrics = {
            "val_acc": self.val_acc(y_hat, y),
            "val_f1_score": f1_score(y_hat, y, task="binary"),
            "val_recall": self.val_recall(y_hat, y),
            "val_precision": self.val_precision(y_hat, y),
            "val_auroc": auroc(y_hat, y, task="binary")
        }

        for metric_name, metric_value in metrics.items():
            pl_trainer.logger.experiment.add_scalars(f'metrics_by_fold/{metric_name}',
                                                     {f'{pl_module.fold_number}': metric_value},
                                                     global_step=pl_module.current_epoch)

    def teardown(self, pl_trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = 'fit') -> None:
        if pl_module.fold_number == self.fold_numbers - 1:

            hparams_dict = {}
            for k, v in pl_module.hparams.items():
                if type(v) == list:  # convert list to string for visualization purposes
                    v = ' '.join(str(e) for e in v)
                hparams_dict[k] = v

            # hparams_dict["unbalance_ratio"] = pl_module.unbalance_ratio

            reduce_ops = ("mean",)
            glob_path = Path(pl_trainer.log_dir)
            avg_metric_values = {}
            for metric in ['f1_score', 'recall', 'precision', 'auroc', 'acc']:
                metric_pattern = 'metrics_by_fold_val_' + metric + '*'
                input_event_dirs = [str(pp) for pp in glob_path.glob(metric_pattern)]
                event_metrics = tbr.load_tb_events(input_event_dirs, handle_dup_steps='keep-last')
                avg_metrics = list(list(tbr.reduce_events(event_metrics, reduce_ops)['mean'].values())[0])

                for epoch_number, metric_value in enumerate(avg_metrics):
                    pl_trainer.logger.experiment.add_scalars(f'metrics_averaged/{metric}', {metric: metric_value},
                                                             global_step=epoch_number)
                avg_metric_values[metric] = avg_metrics

            max_auroc_index = np.argmax(avg_metric_values['auroc'])

            metric_dict = {
                f'hp/max_{metric_name}': metric_values[max_auroc_index]
                for metric_name, metric_values in avg_metric_values.items()
            }

            pl_trainer.logger.experiment.add_hparams(hparams_dict, metric_dict, run_name="best_metric_scores")

