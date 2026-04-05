import argparse
import os
import torch
import pytorch_lightning as pl
import random
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from datasets.leicester_dataset import LeicesterDataset
from models.factory import build_model
from trainers.lightning_module import EEGTrainer
from torch.utils.data import random_split
from torchinfo import summary
import pandas as pd
import json
from datetime import datetime
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", required=True)
    p.add_argument("--train_dirs", nargs="+", required=True)
    p.add_argument("--test_dirs", nargs="*", default=[])

    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--norm",
        type=str,
        default="none",
        choices=["none", "batch", "layer", "group", "instance", "switch"]
    )

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

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--model", required=True)
    p.add_argument("--cnn_channels", nargs="+", type=int, default=[32,64,128])
    p.add_argument("--rnn_hidden", type=int, default=64)

    p.add_argument("--rnn_type", default="lstm")
    p.add_argument("--dense_hidden", type=int, default=64)
    p.add_argument("--use_global_pool", choices=["true","false"], default="false")

    return p.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):


    set_seed(args.seed)
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    run_name = f"{args.model}_ws{args.window_size}_ov{args.overlap}_norm{args.norm}_seed{args.seed}"
    log_dir = os.path.join("logs", args.model, run_id+"_"+run_name)
    os.makedirs(log_dir, exist_ok=True)
    tensorBoard_dir = os.path.join("tensorboard", args.model, run_id+"_"+run_name)
    os.makedirs(tensorBoard_dir, exist_ok=True)


    # save config
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # ===============================
    # SPLIT LOGIC
    # ===============================

    if args.split_mode == "folder":

        train_ds = LeicesterDataset(
            args.data_root,
            args.train_dirs,
            window_size=args.window_size,
            overlap=args.overlap,
            split_mode="folder",
            split_ratio=args.split_ratio,
            split_part="train",
            seed=args.seed   # 👈 thêm
        )

        test_ds = LeicesterDataset(
            args.data_root,
            args.train_dirs,
            window_size=args.window_size,
            overlap=args.overlap,
            split_mode="folder",
            split_ratio=args.split_ratio,
            split_part="test",
            seed=args.seed   # 👈 thêm
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
            split_part="train",
            seed=args.seed   # 👈 thêm
        )

        test_ds = LeicesterDataset(
            args.data_root,
            args.train_dirs,
            window_size=args.window_size,
            overlap=args.overlap,
            split_mode="random_epoch",
            split_ratio=args.split_ratio,
            split_part="test",
            seed=args.seed   # 👈 thêm
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

    val_ratio = 0.1
    val_size = int(len(train_ds) * val_ratio)
    train_size = len(train_ds) - val_size

    train_ds, val_ds = random_split(
        train_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=2,
        persistent_workers=True
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
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
            "rnn_hidden": args.rnn_hidden,
            "rnn_layers": args.rnn_layers,
            "rnn_type": args.rnn_type,
            "dropout": args.dropout,
            "norm": args.norm,
            "dense_hidden": args.dense_hidden,
            "use_global_pool": args.use_global_pool=="true"
        }
    }

    model = build_model(cfg)

    # summary(
    #     model,
    #     input_size=(1,1,args.window_size),
    #     col_names=["input_size", "output_size", "num_params"]
    # )


    lit_model = EEGTrainer(
        model=model, 
        lr=args.lr,
        model_name=args.model,
        run_name=run_id + "_" + run_name
    )

    # ===============================
    # LOGGING + CHECKPOINT
    # ===============================

    ckpt_dir = os.path.join("checkpoints", args.model, run_id + "_" + run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_logger = CSVLogger(save_dir="logs", name=args.model,version=run_id + "_" + run_name)
    tb_logger = TensorBoardLogger(save_dir="tensorboard", name=args.model, version=run_id + "_" + run_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val_acc",     # 🔥 chọn metric chính
        mode="max",
        save_top_k=1,
        save_last=True         # 🔥 sẽ tạo last.ckpt
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=[csv_logger, tb_logger],
        deterministic="warn",
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, early_stop],
        gradient_clip_val=1.0
    )

    trainer.fit(lit_model, train_loader, val_loader)
    # val_results = trainer.test(lit_model, val_loader)
    # test_results = trainer.test(lit_model, test_loader)
    metrics = trainer.callback_metrics

    # lấy history từ CSVLogger
    log_dir = csv_logger.log_dir
    metrics_file = os.path.join(log_dir, "metrics.csv")

    import pandas as pd
    df = pd.read_csv(metrics_file)

    with open(os.path.join(log_dir, "curves.pkl"), "wb") as f:
        pickle.dump(df.to_dict(), f)

    # load best model
    best_path = checkpoint_callback.best_model_path

    print("Best checkpoint:", best_path)
    val_results = trainer.test(lit_model, dataloaders=val_loader, ckpt_path=best_path)
    test_results = trainer.test(lit_model, dataloaders=test_loader, ckpt_path=best_path)


    val_metrics = val_results[0]
    test_metrics = test_results[0]

    metrics = trainer.callback_metrics

    df = pd.DataFrame([{
        "model": args.model,
        "seed": args.seed,
        "window_size": args.window_size,
        "overlap": args.overlap,
        "norm": args.norm,

        "test_acc": metrics["test_acc"].item(),
        "test_bal_acc": metrics["test_bal_acc"].item(),
        "test_f1": metrics["test_f1"].item(),
        "test_precision": metrics["test_precision"].item(),
        "test_recall": metrics["test_recall"].item(),
        "test_kappa": metrics["test_kappa"].item(),
    }])

    save_path = os.path.join("logs", args.model,run_id+"_"+run_name)
    os.makedirs(save_path, exist_ok=True)

    df.to_csv(os.path.join(save_path, "test_results.csv"), index=False)





if __name__ == "__main__":
    args = parse_args()
    main(args)