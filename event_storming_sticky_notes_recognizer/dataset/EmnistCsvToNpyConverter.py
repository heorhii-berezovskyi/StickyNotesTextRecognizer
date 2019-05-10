import argparse
import os

import numpy as np
from numpy import ndarray


class EmnistCsvToNpyConverter:
    """
     Converts emnist data set from csv to .npy format.
    """

    def __init__(self, num_of_channels: int, image_size: int):
        self.num_of_channels = num_of_channels
        self.image_size = image_size

    def to_npy(self, dataset_path: str, new_dir: str, split_name: str, skip_header: int):
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
    def _load(path_from: str, skip_header: int) -> ndarray:
        return np.genfromtxt(path_from, delimiter=',', skip_header=skip_header, dtype=np.uint8)

    @staticmethod
    def _split_into_labels_and_data(dataset: ndarray) -> (ndarray, ndarray):
        labels = dataset[:, 0]
        data = dataset[:, 1:]
        return labels, data


def run(args):
    formatter = EmnistCsvToNpyConverter(num_of_channels=args.num_channels,
                                        image_size=args.image_size)

    formatter.to_npy(dataset_path=args.dataset,
                     new_dir=args.write_to,
                     split_name=args.set_name,
                     skip_header=args.skip_header)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model with specified parameters.')
    parser.add_argument('--num_channels', type=int, help='Number of channels in images.', default=1)
    parser.add_argument('--image_size', type=int, help='Size of the image.', default=28)

    parser.add_argument('--dataset', type=str, help='Path to dataset file in csv format.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\emnist\emnist-balanced-train.csv')

    parser.add_argument('--write_to', type=str, help='Directory to save extracted labels and data.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\emnist_balanced')

    parser.add_argument('--set_name', type=str, help='Type of the dataset(train, test).',
                        default='train')

    parser.add_argument('--skip_header', type=int, help='Number of rows to skip in the dataset',
                        default=0)

    _args = parser.parse_args()
    run(_args)
