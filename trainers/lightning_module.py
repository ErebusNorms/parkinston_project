import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy


class EEGTrainer(pl.LightningModule):

    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

        self.loss_fn = nn.CrossEntropyLoss()

        # metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

    def forward(self, x):
        return self.model(x)

    # ---------------------
    # TRAIN
    # ---------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    # ---------------------
    # VALIDATION
    # ---------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    # ---------------------
    # TEST
    # ---------------------
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)

        self.log("test_acc", acc, prog_bar=True)

    # ---------------------
    # OPTIMIZER
    # ---------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer