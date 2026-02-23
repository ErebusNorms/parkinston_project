from models.cnn import CNN
from models.lstm import LSTM
from models.gru import GRU
from models.cnn_lstm import CNN_LSTM
from models.cnn_gru import CNN_GRU
from models.cnn_bilstm import CNN_BiLSTM
from models.cnn_bilstm_att import CNN_BiLSTM_Att
from models.tcn import TCN


def build_model(cfg):

    name = cfg["model"]["name"]
    ch = cfg["model"]["cnn_channels"]
    h = cfg["model"]["rnn_hidden"]
    layers = cfg["model"]["rnn_layers"]
    dropout = cfg["model"]["dropout"]

    models = {
        "cnn": CNN,
        "lstm": LSTM,
        "gru": GRU,
        "cnn_lstm": CNN_LSTM,
        "cnn_gru": CNN_GRU,
        "cnn_bilstm": CNN_BiLSTM,
        "cnn_bilstm_att": CNN_BiLSTM_Att,
        "tcn": TCN
    }

    if name not in models:
        raise ValueError("Unknown model")

    return models[name](
        cnn_channels=ch,
        hidden=h,
        rnn_layers=layers,
        dropout=dropout
    )