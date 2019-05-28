import json
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from event_storming_sticky_notes_recognizer.Name import Name
from event_storming_sticky_notes_recognizer.dataset.ImageUtils import image_resize
from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder


class TestWordsDataset(Dataset):
    def __init__(self, data_set_path: str, transform=None, alphabet='russian'):
        self.mapping = self._load_file(path=os.path.join(data_set_path, 'mapping.json'))
        self.page = cv2.imread(os.path.join(data_set_path, 'page.tif'), cv2.IMREAD_COLOR)
        self.transform = transform
        self.encoder_decoder = LabelEncoderDecoder(alphabet=alphabet)

    @staticmethod
    def _load_file(path: str):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.mapping['outputs']['object'])

    def __getitem__(self, idx):
        data = self.mapping['outputs']['object']
        x_min = data[idx]['bndbox']['xmin']
        y_min = data[idx]['bndbox']['ymin']
        x_max = data[idx]['bndbox']['xmax']
        y_max = data[idx]['bndbox']['ymax']

        label = data[idx]['name']
        label = self.encoder_decoder.encode_word(word=label)

        image = self.page[y_min: y_max, x_min: x_max, :]
        image = image_resize(image, height=54)

        image_height = image.shape[0]
        image_width = image.shape[1]

        result = np.ones((64, 512, 3), dtype=np.uint8) * 255

        result[int((64 - image_height) / 2): image_height + int((64 - image_height) / 2), 0: image_width, :] = image
        image = result

        image = image.transpose(2, 0, 1)

        sample = {Name.LABEL.value: label.astype(int),
                  Name.IMAGE.value: image,
                  Name.LABEL_LEN.value: self.encoder_decoder.decode_word_len(array=label)}

        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    json_file_path = r'D:\russian_words\real\outputs\SMinolta_Co19052713040.json'
    with open(json_file_path, encoding='utf-8') as f:
        data = json.load(f)
    print()
