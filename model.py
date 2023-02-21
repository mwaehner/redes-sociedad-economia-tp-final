import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss
from torchmetrics import Accuracy, Recall, Precision
from torchmetrics.functional import f1_score


class Model(pl.LightningModule):
    
    def __init__(self,
                 use_cpu: bool = False,
                 learning_rate: float = 0.001,  # 1e-5
                 input_size: int = 43,
                 hidden_size: int = 10,
                 output_size: int = 1,
                 dropout_rate: float = 0.0
                 ):
        super().__init__()
        self.save_hyperparameters()

        # Architecture layers
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate and dropout_rate else None


        self.learning_rate = learning_rate
        self.fold_number: int = 0

        self.all_validation_ys = torch.empty(0, dtype=torch.float32)
        self.all_validation_y_hats = torch.empty(0, dtype=torch.float32)
        self.gpu_used = not use_cpu and torch.cuda.is_available()
        if self.gpu_used:
            self.all_validation_ys = self.all_validation_ys.cuda()
            self.all_validation_y_hats = self.all_validation_y_hats.cuda()

        self.train_acc = Accuracy("binary")
        self.val_acc = Accuracy("binary")
        self.val_recall = Recall("binary")
        self.val_precision = Precision("binary")

        self.criterion = BCEWithLogitsLoss()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        if self.dropout:
            out = self.dropout(out)
        # no activation and no softmax at the end
        return out

    def set_fold(self, fold_number):
        self.fold_number = fold_number

    def validation_epoch_end(self, outputs):
        # Save all validation outputs for metrics logging in callbacks
        ys = [batch_result[0] for batch_result in outputs]
        y_hats = [batch_result[1] for batch_result in outputs]
        self.all_validation_ys = torch.cat(ys).view(-1)
        self.all_validation_y_hats = torch.cat(y_hats).view(-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch.values()
        y_hat = self(x)
        loss = self.criterion(y_hat, y.view(-1,1))
        # self.log_train_metrics(y, y_hat)  # Uncomment for debugging
        self.log(f'fold_{self.fold_number}_loss/training', loss, on_step=False, on_epoch=True)
        return loss

    def log_train_metrics(self, y, y_hat):
        # We log these metrics mostly to see that the training is going alright
        train_acc = self.train_acc(y_hat, y.int())
        train_f1_score = f1_score(y_hat, y.int())
        self.log(f'fold_{self.fold_number}_metrics/train_acc', train_acc, on_step=True, on_epoch=True)
        self.log(f'fold_{self.fold_number}_metrics/train_f1_score', train_f1_score)

    def validation_step(self, batch, batch_idx):
        x, y = batch.values()
        y_hat = self(x)
        loss = self.criterion(y_hat,  y.view(-1,1))
        self.log(f'fold_{self.fold_number}_loss/validation', loss)
        return y, y_hat
