import argparse
import glob
import os

import cv2
import numpy as np
from numpy import ndarray


class LettersToNpyFormatter:
    def __init__(self, parent_dir: str):
        self.directory = parent_dir
        self.mapping = \
            {0: 10, 1: 36, 2: 11, 3: 37, 4: 12, 5: 12, 6: 13, 7: 38,
             8: 14, 9: 39, 10: 15, 11: 40, 12: 16, 13: 41, 14: 17, 15: 42,
             16: 18, 17: 18, 18: 19, 19: 19, 20: 20, 21: 20, 22: 21, 23: 21,
             24: 22, 25: 22, 26: 23, 27: 43, 28: 24, 29: 24, 30: 25, 31: 25,
             32: 26, 33: 44, 34: 27, 35: 45, 36: 28, 37: 28, 38: 29, 39: 48,
             40: 30, 41: 30, 42: 31, 43: 31, 44: 32, 45: 32, 46: 33, 47: 33,
             48: 34, 49: 34, 50: 35, 51: 35}

    def _get_labels_and_images(self, image_paths: list) -> (list, list):
        labels = []
        images = []
        for path in image_paths:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            image_index = int(os.path.splitext(os.path.basename(path))[0])
            label = self._to_label(index=image_index)
            labels.append(label)
        return labels, images

    def _content(self, img_dir: str) -> list:
        return glob.glob(os.path.join(self.directory, img_dir) + r'\*')

    def _to_label(self, index: int) -> int:
        return self.mapping[index]

    def format(self) -> (ndarray, ndarray):
        labels = []
        images = []
        sub_dirs = os.listdir(self.directory)
        for sub_dir in sub_dirs:
            img_dir = os.path.join(self.directory, sub_dir)
            image_paths = self._content(img_dir=img_dir)
            folder_labels, folder_images = self._get_labels_and_images(image_paths=image_paths)
            labels += folder_labels
            images += folder_images
        return np.asarray(labels, dtype=np.uint8), np.asarray(images, dtype=np.uint8)


def run(args):
    formatter = LettersToNpyFormatter(parent_dir=args.directory)
    labels_array, images_array = formatter.format()
    np.save(os.path.join(args.write_to, 'labels.npy'), labels_array)
    np.save(os.path.join(args.write_to, 'images.npy'), images_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Converts images from a specified directory of all folders into labels and data in .npy format.')
    parser.add_argument('--directory', type=str, help='Directory with folders containing images.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset')
    parser.add_argument('--write_to', type=str, help='Path to a directory to save labels and data.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\emnist_balanced')

    _args = parser.parse_args()
    run(args=_args)
