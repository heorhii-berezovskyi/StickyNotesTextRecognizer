import os

import numpy as np
from torch.utils.data import Dataset

from event_storming_sticky_notes_recognizer.Name import Name
from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder


class WordsDataset(Dataset):
    def __init__(self, data_set_dir: str, transform=None):
        self.data = np.load(os.path.join(data_set_dir, 'data.npy'))
        self.labels = np.load(os.path.join(data_set_dir, 'labels.npy'))
        self.transform = transform
        self.encoder_decoder = LabelEncoderDecoder()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = image.reshape(1, 64, 512)
        label = self.labels[idx]

        sample = {Name.LABEL.value: label,
                  Name.IMAGE.value: image,
                  Name.LABEL_LEN.value: self.encoder_decoder.decode_word_len(array=label)}

        if self.transform:
            sample = self.transform(sample)
        return sample
