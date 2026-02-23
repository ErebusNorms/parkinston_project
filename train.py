import argparse
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from datasets.leicester_dataset import LeicesterDataset
from models.factory import build_model
from trainers.lightning_module import EEGTrainer


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", required=True)
    p.add_argument("--train_dirs", nargs="+", required=True)
    p.add_argument("--test_dirs", nargs="*", default=[])

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)

    p.add_argument("--rnn_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--split_mode",
                   type=str,
                   default="folder",
                   choices=["folder", "random_epoch"])

    p.add_argument("--split_ratio", type=float, default=0.8)

    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--overlap", type=float, default=0.25)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--model", required=True)
    p.add_argument("--cnn_channels", nargs="+", type=int, default=[32,64,128])
    p.add_argument("--rnn_hidden", type=int, default=64)

    return p.parse_args()


def main(args):

    # ===============================
    # SPLIT LOGIC
    # ===============================

    if args.split_mode == "folder":

        train_ds = LeicesterDataset(
            args.data_root,
            args.train_dirs,
            window_size=args.window_size,
            overlap=args.overlap,
            split_mode="folder"
        )

        test_ds = LeicesterDataset(
            args.data_root,
            args.test_dirs,
            window_size=args.window_size,
            overlap=args.overlap,
            split_mode="folder"
        )

    elif args.split_mode == "random_epoch":

        print("Using epoch-level random split")

        train_ds = LeicesterDataset(
            args.data_root,
            args.train_dirs,
            window_size=args.window_size,
            overlap=args.overlap,
            split_mode="random_epoch",
            split_ratio=args.split_ratio,
            split_part="train"
        )

        test_ds = LeicesterDataset(
            args.data_root,
            args.train_dirs,
            window_size=args.window_size,
            overlap=args.overlap,
            split_mode="random_epoch",
            split_ratio=args.split_ratio,
            split_part="test"
        )

    else:
        raise ValueError("Unknown split_mode")

    print("Train size:", len(train_ds))
    print("Test size:", len(test_ds))

    if len(train_ds) == 0:
        raise ValueError("Train dataset empty")

    # ===============================
    # DATALOADER
    # ===============================

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=2,
        persistent_workers=True
    )

    # ===============================
    # MODEL
    # ===============================

    cfg = {
        "model": {
            "name": args.model,
            "cnn_channels": args.cnn_channels,
            "rnn_hidden": args.d_model,   # transformer dùng d_model
            "rnn_layers": args.num_layers,
            "dropout": args.dropout,
            "nhead": args.nhead
        }
    }

    model = build_model(cfg)
    lit_model = EEGTrainer(model, lr=args.lr)

    # ===============================
    # LOGGING + CHECKPOINT
    # ===============================

    run_name = f"{args.split_mode}_{'_'.join(args.train_dirs)}"
    ckpt_dir = os.path.join("checkpoints", args.model, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = CSVLogger("logs", name=args.model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop]
    )

    trainer.fit(lit_model, train_loader)
    trainer.test(lit_model, test_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)