import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from event_storming_sticky_notes_recognizer.Name import Name
from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder


class TrainWordsDataset(Dataset):
    def __init__(self, data_set_dir: str, transform=None, alphabet='russian'):
        self.directory = data_set_dir
        self.num_of_pages = len(os.listdir(data_set_dir))
        self.transform = transform
        self.encoder_decoder = LabelEncoderDecoder(alphabet=alphabet)

    def __len__(self):
        return self.num_of_pages

    def __getitem__(self, idx):
        random_page_folder = np.random.randint(self.num_of_pages)
        folder_path = os.path.join(self.directory, str(random_page_folder))
        page = cv2.imread(os.path.join(folder_path, 'page.png'), cv2.IMREAD_COLOR)
        page_labels = np.load(os.path.join(folder_path, 'labels.npy'))
        random_word_index = np.random.randint(len(page_labels))
        word_index = page_labels[random_word_index]
        label = word_index[:16]
        coords = word_index[16:]
        min_h = coords[0]
        max_h = coords[1]
        min_w = coords[2]
        max_w = coords[3]
        image = np.ones((64, 512, 3), dtype=np.uint8) * 255
        image[:, :max_w - min_w, :] = page[min_h: max_h, min_w: max_w, :]
        image = image.transpose(2, 0, 1)

        sample = {Name.LABEL.value: label.astype(int),
                  Name.IMAGE.value: image,
                  Name.LABEL_LEN.value: self.encoder_decoder.decode_word_len(array=label)}

        if self.transform:
            sample = self.transform(sample)
        return sample
