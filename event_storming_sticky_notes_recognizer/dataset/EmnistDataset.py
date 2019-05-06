import numpy as np
from torch.utils.data import Dataset

from event_storming_sticky_notes_recognizer.Name import Name


class EmnistDataset(Dataset):
    def __init__(self, data_path: str, labels_path: str, transform=None):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        sample = {Name.LABEL: label, Name.IMAGE: image}

        if self.transform:
            sample = self.transform(sample)

        return sample
