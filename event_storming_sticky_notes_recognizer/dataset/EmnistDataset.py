import os

import numpy as np
from torch.utils.data import Dataset

from event_storming_sticky_notes_recognizer.Name import Name


class EmnistDataset(Dataset):
    def __init__(self, data_set_dir: str, data_set_type: str, transform=None):
        self.data = np.load(os.path.join(data_set_dir, data_set_type + '_data.npy'))
        self.labels = np.load(os.path.join(data_set_dir, data_set_type + '_labels.npy'))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = image.reshape(1, 64, 512)
        label = self.labels[idx]

        sample = {Name.LABEL.value: label, Name.IMAGE.value: image}

        if self.transform:
            sample = self.transform(sample)

        return sample
