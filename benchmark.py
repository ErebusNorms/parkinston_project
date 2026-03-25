import subprocess
import pandas as pd
import os

# ========================
# CONFIG
# ========================

DATA_ROOT = "data/leicester_dataset"
TRAIN_DIRS = ['A10', 'A11', 'A14', 'A15', 'A16', 'A3', 'A7', 'A8', 'A9', 'B1', 'B11', 'B12','B13', 'B15', 'B16', 'B3', 'B4', 'B6', 'B8', 'B9', 'C1', 'C10', 'C13', 'C14', 'C16', 'C17','C2', 'C3', 'C5', 'C6', 'C8', 'C9']

# ["A9", "B1", "B3", "B6", "C16", "C6", "C8"]

COMMON_ARGS = [
    "--data_root", DATA_ROOT,
    "--train_dirs", *TRAIN_DIRS,
    "--split_mode", "random_epoch",
    "--epochs", "15",
    "--seed", "42"
]

# ========================
# 7 MODELS
# ========================

MODELS = [
    ("cnn", [
        "--model", "cnn",
        "--cnn_channels", "32", "64", "128",
        "--use_global_pool", "True"
    ]),

    ("lstm", [
        "--model", "lstm",
        "--rnn_type", "lstm",
        "--rnn_layers", "3"
    ]),

    ("gru", [
        "--model", "gru",
        "--rnn_type", "gru",
        "--rnn_layers", "3"
    ]),

    ("bilstm", [
        "--model", "bilstm",
        "--rnn_type", "bilstm",
        "--rnn_layers", "3"
    ]),

    ("cnn_lstm", [
        "--model", "cnn_lstm",
        "--cnn_channels", "32", "64", "128",
        "--rnn_type", "lstm",
        "--rnn_layers", "1"
    ]),

    ("cnn_gru", [
        "--model", "cnn_gru",
        "--cnn_channels", "32", "64", "128",
        "--rnn_type", "gru",
        "--rnn_layers", "1"
    ]),

    ("cnn_bilstm", [
        "--model", "cnn_bilstm",
        "--cnn_channels", "32", "64", "128",
        "--rnn_type", "bilstm",
        "--rnn_layers", "1"
    ]),
]

# ========================
# RUN ALL
# ========================

results = []

for name, model_args in MODELS:

    print(f"\n===== RUNNING {name.upper()} =====\n")

    cmd = ["python", "train.py"] + COMMON_ARGS + model_args

    subprocess.run(cmd)

    # đọc test result
    path = f"logs/{name}/test_results.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        row = df.iloc[0].to_dict()
        row["model"] = name
        results.append(row)

# ========================
# SAVE FINAL RESULT
# ========================

final_df = pd.DataFrame(results)

final_df = final_df[[
    "model",
    "test_acc",
    "test_bal_acc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_kappa"
]]

final_df = final_df.sort_values("test_f1", ascending=False)

final_df.to_csv("benchmark_results.csv", index=False)

print("\n===== FINAL RESULT =====\n")
print(final_df)