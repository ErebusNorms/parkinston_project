import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


ACTIVE_GROUP = {
    "A2","A4","A6","A7","A11","A12","A13","A15","A16",
    "B4","B17","B9","B10","B12","B16",
    "C1","C2","C4","C6","C8","C11","C14","C15","C16"
}

SHAM_GROUP = {
    "A1","A3","A5","A8","A9","A10","A14",
    "B1","B2","B3","B5","B6","B8","B11","B13","B14","B15",
    "C3","C5","C7","C9","C10","C12","C13","C17"
}


class LeicesterDataset(Dataset):

    def __init__(self,
                 root_dir,
                 selected_dirs,
                 window_size=64,
                 overlap=0.25,
                 split_mode="folder",
                 split_ratio=0.8,
                 split_part="train"):  # train | test

        self.window_size = window_size
        self.step = int(window_size * (1 - overlap))

        self.index_map = []
        self.file_labels = {}

        for folder in selected_dirs:

            folder_path = os.path.join(root_dir, folder)

            if not os.path.exists(folder_path):
                continue

            # label
            if folder in ACTIVE_GROUP:
                label = 1
            elif folder in SHAM_GROUP:
                label = 0
            else:
                continue

            for f in os.listdir(folder_path):

                if "." in f:
                    continue

                file_path = os.path.join(folder_path, f)

                with open(file_path, "rb") as fp:
                    data = pickle.load(fp)

                data = np.array(data)

                # handle shape
                if data.shape == (520, 512):
                    num_epochs = 520
                    epoch_axis = 0
                elif data.shape == (512, 520):
                    num_epochs = 520
                    epoch_axis = 1
                else:
                    continue

                self.file_labels[file_path] = label

                # -------- Epoch-level split --------
                epoch_indices = list(range(num_epochs))

                if split_mode == "random_epoch":

                    np.random.seed(42)
                    np.random.shuffle(epoch_indices)

                    split_point = int(len(epoch_indices) * split_ratio)

                    if split_part == "train":
                        epoch_indices = epoch_indices[:split_point]
                    else:
                        epoch_indices = epoch_indices[split_point:]

                # -------- Sliding window --------
                for epoch_idx in epoch_indices:

                    if epoch_axis == 0:
                        signal = data[epoch_idx]
                    else:
                        signal = data[:, epoch_idx]

                    for start in range(
                        0,
                        512 - window_size + 1,
                        self.step
                    ):
                        self.index_map.append(
                            (file_path, epoch_idx, start)
                        )

        print("Total windows:", len(self.index_map))

        self._cache_file = None
        self._cache_data = None

    def __len__(self):
        return len(self.index_map)

    def _load_file(self, file_path):
        if self._cache_file != file_path:
            with open(file_path, "rb") as fp:
                data = pickle.load(fp)
            self._cache_data = np.array(data)
            self._cache_file = file_path
        return self._cache_data

    def __getitem__(self, idx):

        file_path, epoch_idx, start = self.index_map[idx]

        data = self._load_file(file_path)

        if data.shape == (520, 512):
            signal = data[epoch_idx]
        else:
            signal = data[:, epoch_idx]

        label = self.file_labels[file_path]
        x = signal[start:start+self.window_size]

        return (
            torch.tensor(x).unsqueeze(0).float(),
            torch.tensor(label).long()
        )