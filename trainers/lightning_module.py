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

from sklearn.metrics import (
    cohen_kappa_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


class EEGTrainer(pl.LightningModule):

    def __init__(self, model, lr=1e-3, model_name="model", run_name="run"):
        super().__init__()

        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

        self.save_hyperparameters()

        # ===== train =====
        self.train_acc = BinaryAccuracy()

        # ===== val =====
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

        self.log("train_loss", loss,
                 on_step=False, on_epoch=True,
                 prog_bar=True,
                 batch_size=x.size(0))

        self.log("train_acc", self.train_acc,
                 on_step=False, on_epoch=True,
                 prog_bar=True,
                 batch_size=x.size(0))

        return loss

    # ================= VALID =================
    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        if len(y) > 0:
            self.val_acc.update(preds, y)
            self.val_auc.update(probs, y)
            self.val_f1.update(preds, y)
            self.val_precision.update(preds, y)
            self.val_recall.update(preds, y)

        self.log("val_loss", loss,
                 on_step=False, on_epoch=True,
                 prog_bar=True,
                 batch_size=x.size(0))

        self.log("val_acc", self.val_acc,
                 on_step=False, on_epoch=True,
                 prog_bar=True,
                 batch_size=x.size(0))

        self.log("val_auc", self.val_auc,
                 on_step=False, on_epoch=True,
                 batch_size=x.size(0))

        self.log("val_f1", self.val_f1,
                 on_step=False, on_epoch=True,
                 batch_size=x.size(0))

        self.log("val_precision", self.val_precision,
                 on_step=False, on_epoch=True,
                 batch_size=x.size(0))

        self.log("val_recall", self.val_recall,
                 on_step=False, on_epoch=True,
                 batch_size=x.size(0))

    # ================= TEST =================
    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []

    def on_test_epoch_end(self):

        preds = torch.cat(self.test_preds).numpy()
        targets = torch.cat(self.test_targets).numpy()

        # ===== metrics =====
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

        # ===== SAVE PATH =====
        save_dir = os.path.join(
            "logs",
            self.hparams.model_name,
            self.hparams.run_name
        )
        os.makedirs(save_dir, exist_ok=True)

        # ===== classification report =====
        report = classification_report(
            targets,
            preds,
            output_dict=True,
            zero_division=0
        )

        pd.DataFrame(report).transpose().to_csv(
            os.path.join(save_dir, "classification_report.csv")
        )

        # ===== confusion matrix =====
        cm = confusion_matrix(targets, preds)

        np.save(os.path.join(save_dir, "confusion_matrix.npy"), cm)

        # ===== plot =====
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()

        print("\nConfusion Matrix:\n", cm)
        print("\nUnique preds:", np.unique(preds))

    # ================= OPT =================
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ================= RESET =================
    def on_train_epoch_end(self):
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.val_acc.reset()
        self.val_auc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()