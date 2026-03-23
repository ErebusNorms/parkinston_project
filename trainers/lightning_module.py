import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall
)

from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score


class EEGTrainer(pl.LightningModule):

    def __init__(self, model, lr=1e-3):
        super().__init__()

        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

        # ===== train metrics =====
        self.train_acc = BinaryAccuracy()

        # ===== val metrics =====
        self.val_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()

        # ===== test storage =====
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    # ================= TRAIN =================
    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)

        self.train_acc.update(preds, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", self.train_acc, on_epoch=True)

        return loss

    # ================= VALID =================
    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, y)
        self.val_auc.update(probs, y)
        self.val_f1.update(preds, y)
        self.val_precision.update(preds, y)
        self.val_recall.update(preds, y)

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", self.val_acc, on_epoch=True)
        self.log("val_auc", self.val_auc, on_epoch=True)
        self.log("val_f1", self.val_f1, on_epoch=True)
        self.log("val_precision", self.val_precision, on_epoch=True)
        self.log("val_recall", self.val_recall, on_epoch=True)

    # ================= TEST =================
    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())

    def on_test_epoch_end(self):

        preds = torch.cat(self.test_preds).numpy()
        targets = torch.cat(self.test_targets).numpy()

        acc = (preds == targets).mean()
        bal_acc = balanced_accuracy_score(targets, preds)
        f1 = BinaryF1Score()(torch.tensor(preds), torch.tensor(targets)).item()
        precision = BinaryPrecision()(torch.tensor(preds), torch.tensor(targets)).item()
        recall = BinaryRecall()(torch.tensor(preds), torch.tensor(targets)).item()
        kappa = cohen_kappa_score(targets, preds)

        self.log_dict({
            "test_acc": acc,
            "test_bal_acc": bal_acc,
            "test_f1": f1,
            "test_precision": precision,
            "test_recall": recall,
            "test_kappa": kappa
        })

    # ================= OPT =================
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)