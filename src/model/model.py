from turtle import forward
from torch import nn
import torch
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl

from torchmetrics import (Accuracy, 
                            Precision, 
                            Recall)


class WoofModel(nn.Module):
    """
    Model, which solves our classification task
    Args:
    backbone: model for feature extraction
    """

    def __init__(self, backbone: nn.Module):
        super(WoofModel, self).__init__()

        num_filters = backbone.num_features
        self.feature_extraction = backbone

        self.softamx = nn.Softmax(dim=-1)
        self.fc = nn.Linear(num_filters, 10)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)

        cls = self.fc(x)
        cls = self.softamx(cls) + 1e-4

        return cls


class WoofTrainModule(pl.LightningModule):
    """
    Module, which train our model and log all metrics
    Args:
    backbone: model for feature extraction
    """

    def __init__(self, backbone, LR=0.001):
        super().__init__()

        self.LR = LR
        self.model = WoofModel(backbone)
        self.loss = nn.CrossEntropyLoss()

        self.train_accuracy =  Accuracy()

        self.valid_accuracy = Accuracy()
        self.valid_precision = Precision()
        self.valid_recall = Recall()

        self.train_metrics = {
            'train_accuracy':  self.train_accuracy,
        }

        self.valid_metrics = {
            'valid_accuracy': self.valid_accuracy,
            'valid_precison': self.valid_precision,
            'valid_recall': self.valid_recall
        }

    def forward(self, x):
        x = self.model(x)
        return x

    def log_metrics(self, name, loss, l, preds, metrics, on_step=True):
        self.log(name, loss, on_epoch=True, prog_bar=True)

        for key, metric in metrics.items():
            v = metric(preds, l)
            self.log(key, v, on_epoch=True, on_step=on_step)

    def configure_optimizers(self):
        params = []
        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value], 'lr': self.LR * 1.0, 'weight_decay': 5e-4}]

        optimizer = torch.optim.SGD(params, momentum=0.9)
        lr_scheduler = StepLR(optimizer, 5, 0.5)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, train_batch, idx):
        x, l = train_batch

        preds = self.model(x)
        loss = self.loss(preds, l)

        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step()
        
        self.log_metrics('train_loss', loss, l, preds, self.train_metrics)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, l = val_batch
        preds = self.model(x)

        loss = self.loss(preds, l)

        self.log_metrics('valid_loss', loss, l, preds, self.valid_metrics, on_step=False)
        return loss


    def backward(self, loss, optimizer, idx):
        loss.backward()