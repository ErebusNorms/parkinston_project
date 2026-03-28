import subprocess
import pandas as pd
import os

# ========================
# CONFIG
# ========================

DATA_ROOT = "data/leicester_dataset"

TRAIN_DIRS = ["A9", "B1", "B3", "B6", "C16", "C6", "C8"]

# ========================
# SEARCH SPACE
# ========================

MODELS = [

    ("cnn", [
        "--model", "cnn",
        "--cnn_channels", "32", "64", "128"
    ]),

    ("lstm", [
        "--model", "lstm",
        "--rnn_type", "lstm",
        "--rnn_hidden", "64",
        "--rnn_layers", "3"
    ]),

    ("gru", [
        "--model", "gru",
        "--rnn_type", "gru",
        "--rnn_hidden", "64",
        "--rnn_layers", "3"
    ]),

    ("bilstm", [
        "--model", "bilstm",
        "--rnn_type", "bilstm",
        "--rnn_hidden", "64",
        "--rnn_layers", "3"
    ]),

    ("cnn_lstm", [
        "--model", "cnn_lstm",
        "--cnn_channels", "32", "64", "128",
        "--rnn_type", "lstm",
        "--rnn_hidden", "64",
        "--rnn_layers", "1"
    ]),

    ("cnn_gru", [
        "--model", "cnn_gru",
        "--cnn_channels", "32", "64", "128",
        "--rnn_type", "gru",
        "--rnn_hidden", "64",
        "--rnn_layers", "1"
    ]),

    ("cnn_bilstm", [
        "--model", "cnn_bilstm",
        "--cnn_channels", "32", "64", "128",
        "--rnn_type", "bilstm",
        "--rnn_hidden", "64",
        "--rnn_layers", "1"
    ]),
]

NORMS = ["none", "batch", "layer"] # ["none", "batch", "layer", "group", "instance", "switch"]
SEEDS = [42, 12, 34]
WINDOW_SIZES = [64, 128, 256]
OVERLAPS = [0.5]
BATCH_SIZE = [512]
EPOCHS = 50


# ========================
# RUN
# ========================

results = []

for seed in SEEDS:
    for norm in NORMS:
        for ws in WINDOW_SIZES:
            for ov in OVERLAPS:
                for bs in BATCH_SIZE:
                    for model_name, model_args in MODELS:

                        print(f"\n===== RUN =====")
                        print(f"{model_name} | norm={norm} | seed={seed} | ws={ws} | ov={ov}")

                        cmd = [
                            "python", "train.py",
                            "--data_root", DATA_ROOT,
                            "--train_dirs", *TRAIN_DIRS,
                            "--split_mode", "random_epoch",
                            "--epochs", str(EPOCHS),
                            "--batch_size", str(bs),
                            "--seed", str(seed),
                            "--norm", norm,
                            "--window_size", str(ws),
                            "--overlap", str(ov),
                        ] + model_args

                        subprocess.run(cmd)

                        # ===== load result =====
                        path = f"logs/{model_name}/test_results.csv"

                        if os.path.exists(path):
                            df = pd.read_csv(path)
                            row = df.iloc[0].to_dict()

                            row.update({
                                "model": model_name,
                                "norm": norm,
                                "seed": seed,
                                "window_size": ws,
                                "overlap": ov
                            })

                            results.append(row)

# ========================
# RAW RESULT
# ========================

df = pd.DataFrame(results)

df.to_csv("benchmark_raw.csv", index=False)

print("\n===== RAW DONE =====")

# ========================
# SUMMARY (GROUP BY CONFIG)
# ========================

group_cols = ["model", "norm", "window_size", "overlap"]

summary = df.groupby(group_cols).agg({
    "test_acc": ["mean", "std"],
    "test_bal_acc": ["mean", "std"],
    "test_f1": ["mean", "std"],
    "test_precision": ["mean", "std"],
    "test_recall": ["mean", "std"],
    "test_kappa": ["mean", "std"]
}).reset_index()

summary.columns = [
    "model", "norm", "window_size", "overlap",
    "acc_mean", "acc_std",
    "bal_acc_mean", "bal_acc_std",
    "f1_mean", "f1_std",
    "precision_mean", "precision_std",
    "recall_mean", "recall_std",
    "kappa_mean", "kappa_std"
]

summary = summary.sort_values("f1_mean", ascending=False)

summary.to_csv("benchmark_summary.csv", index=False)

print("\n===== SUMMARY =====")
print(summary.head(10))

# ========================
# BEST CONFIG
# ========================

best = summary.iloc[0]

print("\n🏆 BEST CONFIG:")
print(best)

# ========================
# BEST PER MODEL
# ========================

best_per_model = summary.loc[
    summary.groupby("model")["f1_mean"].idxmax()
]

best_per_model.to_csv("benchmark_best_per_model.csv", index=False)

print("\n===== BEST PER MODEL =====")
print(best_per_model)