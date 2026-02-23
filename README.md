# 🧠 Parkinson EEG Classification Framework

This project provides a deep learning framework for EEG-based Parkinson
vs Sham classification.\
It supports multiple neural architectures including CNN, LSTM, GRU,
BiLSTM, Attention-based models, and TCN.

The framework is designed for:

-   Cross-subject evaluation
-   Within-subject (epoch-level) evaluation
-   Flexible model configuration via command line
-   Reproducible experiments using Conda environments
-   Automatic logging and checkpointing

------------------------------------------------------------------------

# 🔬 Project Overview

The system processes EEG epochs (Cz channel) and applies sliding window
segmentation before feeding them into configurable deep learning
architectures.

Each EEG file is expected to have shape:

    (520, 512)

-   520 epochs
-   512 samples per epoch
-   0.5 second window per epoch

------------------------------------------------------------------------

# 🏗️ Available Models

The framework supports the following architectures:

-   `cnn`
-   `lstm`
-   `gru`
-   `cnn_lstm`
-   `cnn_gru`
-   `cnn_bilstm`
-   `cnn_bilstm_att`
-   `tcn`

All models support configurable complexity parameters such as:

-   CNN depth
-   Hidden size
-   Number of RNN layers
-   Dropout rate

------------------------------------------------------------------------

# ⚙️ Reproducible Environment

To ensure all team members use the exact same environment, this project
includes an:

    environment.yml

file.

## Using the environment file

After cloning the repository:

``` bash
conda env create -f environment.yml
conda activate eeg_leicester
```

This guarantees consistent dependency versions across machines.

If the environment needs updating, regenerate it on the main machine:

``` bash
conda activate eeg_leicester
conda env export > environment.yml
```

------------------------------------------------------------------------

# 🚀 Training Usage

The framework is fully configurable from the command line.

## Cross-Subject Evaluation

``` bash
python train.py \
--data_root data/leicester_dataset \
--train_dirs A1 A2 \
--test_dirs B1 B2 \
--split_mode folder \
--model cnn_bilstm_att \
--cnn_channels 64 128 256 \
--rnn_hidden 128 \
--rnn_layers 2 \
--dropout 0.3 \
--epochs 30
```

## Within-Subject (Epoch-Level Split)

``` bash
python train.py \
--data_root data/leicester_dataset \
--train_dirs A1 A2 \
--split_mode random_epoch \
--split_ratio 0.8 \
--model lstm \
--rnn_hidden 128 \
--rnn_layers 2 \
--dropout 0.2 \
--epochs 30
```

------------------------------------------------------------------------

# 🎛️ Key Parameters

  Parameter          Description
  ------------------ -----------------------------------
  ``` bash
  `--model`          Select model architecture
  `--cnn_channels`   CNN layer sizes
  `--rnn_hidden`     Hidden size for RNN layers
  `--rnn_layers`     Number of RNN layers
  `--dropout`        Dropout rate
  `--window_size`    Sliding window size
  `--overlap`        Sliding window overlap ratio
  `--split_mode`     `folder` or `random_epoch`
  `--split_ratio`    Train/Test ratio for random_epoch
  ```
------------------------------------------------------------------------

# 📊 Outputs

During training:

-   Metrics are saved in `/logs`
-   Model checkpoints are saved in `/checkpoints`
-   CSV logs are automatically generated

------------------------------------------------------------------------

# 👥 Contributors

-   Nguyen Van Quan\
-   Research Team Members

------------------------------------------------------------------------

# 📌 Notes

-   `folder` split performs cross-subject evaluation
-   `random_epoch` split prevents window-level leakage
-   Recommended GPU: GTX 1650 Ti or higher
