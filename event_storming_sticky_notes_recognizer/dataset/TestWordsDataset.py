import json
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from event_storming_sticky_notes_recognizer.Name import Name
from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder


class TestWordsDataset(Dataset):
    def __init__(self, data_set_path: str, transform=None, alphabet='russian'):
        self.data = self._load_file(path=data_set_path)
        self.transform = transform
        self.encoder_decoder = LabelEncoderDecoder(alphabet=alphabet)

    @staticmethod
    def _load_file(path: str):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data['outputs']['object'])

    def __getitem__(self, idx):
        page = cv2.imread(self.data['path'], cv2.IMREAD_COLOR)

        data = self.data['outputs']['object']
        x_min = data[idx]['bndbox']['xmin']
        y_min = data[idx]['bndbox']['ymin']
        x_max = data[idx]['bndbox']['xmax']
        y_max = data[idx]['bndbox']['ymax']

        label = data[idx]['name']
        label = self.encoder_decoder.encode_word(word=label)

        image = page[y_min: y_max, x_min: x_max, :]
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


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


if __name__ == "__main__":
    json_file_path = r'D:\russian_words\real\outputs\SMinolta_Co19052713040.json'
    with open(json_file_path, encoding='utf-8') as f:
        data = json.load(f)
    print()
