import argparse
import os

import cv2
import numpy as np
from numpy import ndarray


class DatasetCreator:
    def __init__(self, data_path: str, labels_path: str, words_path: str):
        self.data = np.load(file=data_path)
        self.labels = np.load(file=labels_path)
        self.word_list = self._load(file_path=words_path)
        self.mapping = {'a': 36, 'b': 37, 'c': 12, 'd': 38,
                        'e': 39, 'f': 40, 'g': 41, 'h': 42,
                        'i': 18, 'j': 19, 'k': 20, 'l': 21,
                        'm': 22, 'n': 43, 'o': 24, 'p': 25,
                        'q': 44, 'r': 45, 's': 28, 't': 48,
                        'u': 30, 'v': 31, 'w': 32, 'x': 33,
                        'y': 34, 'z': 35}

    def to_image(self, word: str) -> (ndarray, ndarray):
        word = word.strip()
        word_image = np.zeros((32, 512), dtype=np.uint8)
        word_label = np.zeros(16, dtype=np.uint8)
        num_of_occupied_width_pixels = 0
        for i in range(len(word)):
            letter_class = self.mapping[word[i]]
            corresponding_label_indices = np.where(self.labels == letter_class)
            selected_index = np.random.choice(corresponding_label_indices[0], 1)

            letter_image = self.data[selected_index][0]

            im2, contours, hierarchy = cv2.findContours(letter_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            min_x = min_y = 10000
            max_x = max_y = 0
            for c in contours:
                # Returns the location and width,height for every contour
                x, y, w, h = cv2.boundingRect(c)
                if w * h > 25:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x + w)

                    min_y = min(min_y, y)
                    max_y = max(max_y, y + h)
            letter_image = letter_image[:, min_x:max_x]

            letter_image = np.pad(letter_image, ((2, 2), (1, 1)), 'constant')

            word_image[:, num_of_occupied_width_pixels: num_of_occupied_width_pixels + max_x - min_x + 2] = letter_image

            num_of_occupied_width_pixels = num_of_occupied_width_pixels + max_x - min_x + 2
            word_label[i] = letter_class
        return word_label, word_image

    def create(self, words_count: int) -> (ndarray, ndarray):
        labels = []
        images = []
        for word in self.word_list:
            for c in range(words_count):
                word_label, word_image = self.to_image(word=word)
                labels.append(word_label)
                images.append(word_image)
        return np.asarray(labels), np.asarray(images)

    def decode(self, label: ndarray) -> str:
        result = ''
        for l in label:
            if l == 0:
                return result
            for character, code in self.mapping.items():
                if code == l:
                    result += character
        return result

    @staticmethod
    def _load(file_path: str) -> list:
        with open(file_path) as f:
            result = f.readlines()
        return result


def run(args):
    creator = DatasetCreator(data_path=args.data_path,
                             labels_path=args.labels_path,
                             words_path=args.words_path)

    labels, images = creator.create(words_count=args.words_count)
    np.save(os.path.join(args.save_to, 'labels.npy'), labels)
    np.save(os.path.join(args.save_to, 'images.npy'), images)


def run_labels_check(args):
    creator = DatasetCreator(data_path=args.data_path,
                             labels_path=args.labels_path,
                             words_path=args.words_path)

    labels = np.load(os.path.join(args.save_to, 'labels.npy'))
    images = np.load(os.path.join(args.save_to, 'images.npy'))
    for i in range(200, 250):
        image = images[i]
        print(creator.decode(label=labels[i]))
        image = image.reshape(32, 512)
        cv2.imshow('img', image)
        cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creates words dataset based on letter images.')
    parser.add_argument('--data_path', type=str, help='Path to letter images in a .npy file.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\emnist_balanced\images.npy')

    parser.add_argument('--labels_path', type=str, help='Path to letter labels in a .npy file.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\emnist_balanced\labels.npy')

    parser.add_argument('--words_path', type=str,
                        help='Path to a text file with words based on dataset will be created.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\words\words.txt')

    parser.add_argument('--words_count', type=int,
                        help='Number of word copies to create.',
                        default=2)

    parser.add_argument('--save_to', type=str,
                        help='Path to a directory to save created dataset.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\words')

    parser.add_argument('--max_length', type=int,
                        help='Maximal number of letters in a single word.',
                        default=16)

    parser.add_argument('--letter_size', type=int,
                        help='Value of a single letter bin after centering.',
                        default=32)

    _args = parser.parse_args()

    # run(_args)
    run_labels_check(_args)
