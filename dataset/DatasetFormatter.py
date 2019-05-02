import os

import numpy as np
from numpy import ndarray


class DatasetFormatter:
    """
     Converts data set from csv to .npy format.
    """

    def __init__(self, num_of_channels: int, image_size: int):
        self.num_of_channels = num_of_channels
        self.image_size = image_size

    def to_npy(self, dataset_path: str, new_dir: str, split_name: str, skip_header=0):
        dataset = self._load(path_from=dataset_path, skip_header=skip_header)
        labels, data = self._split_into_labels_and_data(dataset=dataset)

        data = np.apply_along_axis(func1d=self._rotate, arr=data, axis=1)
        data = data.reshape((labels.size, self.num_of_channels, self.image_size, self.image_size))

        np.save(os.path.join(new_dir, (split_name + '_data')), data)
        np.save(os.path.join(new_dir, (split_name + '_labels')), labels)

    def _rotate(self, image) -> ndarray:
        image = image.reshape([self.image_size, self.image_size])
        image = np.fliplr(image)
        image = np.rot90(image)
        return image

    @staticmethod
    def _load(path_from: str, skip_header=0) -> ndarray:
        return np.genfromtxt(path_from, delimiter=',', skip_header=skip_header, dtype=np.uint8)

    @staticmethod
    def _split_into_labels_and_data(dataset: ndarray) -> (ndarray, ndarray):
        labels = dataset[:, 0]
        data = dataset[:, 1:]
        return labels, data
