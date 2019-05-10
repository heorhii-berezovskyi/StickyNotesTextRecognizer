import argparse
import os

import cv2
import numpy as np
from numpy import ndarray

from event_storming_sticky_notes_recognizer.Exception import UnsupportedParamException


class DatasetCreator:
    def __init__(self, dataset: str, words_path: str, pad_value: int, word_height: int, min_letter_size: int,
                 max_letter_size: int, tall_to_low_letter_coef: float):
        self.directory = dataset
        self.word_list = self._load(file_path=words_path)
        self.pad_val = pad_value
        self.word_h = word_height
        self.min_size = min_letter_size
        self.max_size = max_letter_size
        self.coef = tall_to_low_letter_coef

        self.num_copies = self._get_num_of_copies(data_set_dir=dataset)
        self.images_mapping = {'a': 1, 'b': 3, 'c': 5, 'd': 7,
                               'e': 9, 'f': 11, 'g': 13, 'h': 15,
                               'i': 17, 'j': 19, 'k': 21, 'l': 23,
                               'm': 25, 'n': 27, 'o': 29, 'p': 31,
                               'q': 33, 'r': 35, 's': 37, 't': 39,
                               'u': 41, 'v': 43, 'w': 45, 'x': 47,
                               'y': 49, 'z': 51}

    @staticmethod
    def _get_num_of_copies(data_set_dir: str) -> int:
        markers_count = len(os.listdir(os.path.join(data_set_dir, 'marker')))
        pens_count = len(os.listdir(os.path.join(data_set_dir, 'pen')))
        assert markers_count == pens_count
        return markers_count

    def char_to_image(self, character: str, stroke: str):
        char_index = self.images_mapping[character]
        random_folder = np.random.randint(1, self.num_copies + 1)
        image = cv2.imread(os.path.join(self.directory,
                                        stroke,
                                        str(random_folder),
                                        (str(char_index) + '.png')
                                        ),
                           cv2.IMREAD_GRAYSCALE)
        if character in ['b', 'd', 'f', 'h', 'k', 'l', 't']:
            return self.resize_long_letter(image, type='top')
        elif character in ['g', 'j', 'p', 'q', 'y']:
            return self.resize_long_letter(image, type='bot')
        elif character in ['a', 'c', 'e', 'i', 'm', 'n', 'o', 'r', 's', 'u', 'v', 'w', 'x', 'z']:
            return self.resize_short_letter(image)
        else:
            raise UnsupportedParamException('Character ' + character + ' is not supported.')

    def resize_long_letter(self, letter: ndarray, type: str) -> ndarray:
        h, w = letter.shape
        new_h = np.random.randint(self.min_size, self.max_size)
        new_w = int(w * new_h / h)
        resized = cv2.resize(letter, (new_w, new_h))
        padded = self.pad(arr=resized, top=self.pad_val, bot=self.pad_val, left=self.pad_val, right=self.pad_val)
        result = np.zeros((self.word_h, padded.shape[1]))
        if type == 'top':
            result[self.pad_val * 2:padded.shape[0] + self.pad_val * 2, :] = padded
        elif type == 'bot':
            result[self.word_h - padded.shape[0] - self.pad_val * 2: self.word_h - self.pad_val * 2, :] = padded
        else:
            raise UnsupportedParamException('Letter type ' + type + ' is not supported.')
        return result

    def resize_short_letter(self, letter: ndarray) -> ndarray:
        h, w = letter.shape
        new_h = np.random.randint(int(self.min_size / self.coef), int(self.max_size / self.coef))
        new_w = int(w * new_h / h)
        resized = cv2.resize(letter, (new_w, new_h))
        padded = self.pad(arr=resized, top=self.pad_val, bot=self.pad_val, left=self.pad_val, right=self.pad_val)
        result = np.zeros((self.word_h, padded.shape[1]))

        third_random = np.random.randint(int(self.word_h / 3) - 5, int(self.word_h / 3) + 5)
        result[third_random - self.pad_val * 2: third_random + padded.shape[0] - self.pad_val * 2, :] = padded
        return result

    @staticmethod
    def pad(arr: ndarray, top: int, bot: int, left: int, right: int):
        return np.pad(arr, ((top, bot), (left, right)), mode='constant')

    def to_image(self, word: str) -> (ndarray, ndarray):
        word = word.strip()
        word_image = np.zeros((self.word_h, 512), dtype=np.uint8)
        word_label = np.zeros(16, dtype=np.uint8)
        num_of_occupied_width = 0
        for i in range(len(word)):
            letter_class = self.images_mapping[word[i]]
            letter_image = self.char_to_image(character=word[i], stroke='pen')

            word_image[:, num_of_occupied_width: num_of_occupied_width + letter_image.shape[1]] = letter_image

            num_of_occupied_width += letter_image.shape[1]
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
            for character, code in self.images_mapping.items():
                if code == l:
                    result += character
        return result

    @staticmethod
    def _load(file_path: str) -> list:
        with open(file_path) as f:
            result = f.readlines()
        return result

    # @staticmethod
    # def center_frame(images: list) -> list:
    #     squares = []
    #     for image in images:
    #         height, width = image.shape
    #         if height > width:
    #             differ = height
    #         else:
    #             differ = width
    #
    #         mask = np.zeros((differ, differ), dtype=np.uint8)
    #         x_pos = int((differ - width) / 2)
    #         y_pos = int((differ - height) / 2)
    #         mask[y_pos:y_pos + height, x_pos:x_pos + width] = image[0:height, 0:width]
    #         # thresh, img_bin = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY)
    #         squares.append(mask)
    #     return squares
    #
    # @staticmethod
    # def resize_and_resample(images: list, size: int) -> list:
    #     results = []
    #     for image in images:
    #         resized = cv2.resize(image, (size - 4, size - 4), interpolation=cv2.INTER_AREA)
    #         padded = np.pad(resized, ((2, 2), (2, 2)), 'constant')
    #         padded = padded * 1.7
    #         padded[padded > 255.] = 255.
    #         padded = padded.astype(np.uint8)
    #         results.append(padded)
    #     return results


# def run_center_frame(args):
#     converter = DatasetCreator()
#     image_names = glob.glob(args.img_dir)
#     images = converter.read_images(image_names=image_names)
#
#     squares = converter.center_frame(images=images)
#
#     converter.save_letters(letters=squares,
#                            letters_names=image_names,
#                            path_to=args.write_to)
#
#
# def run_resize_resample(args):
#     converter = DatasetCreator()
#     image_names = glob.glob(args.img_dir)
#     images = converter.read_images(image_names=image_names)
#
#     resized_squares = converter.resize_and_resample(images=images,
#                                                     size=args.final_size)
#     converter.save_letters(letters=resized_squares,
#                            letters_names=image_names,
#                            path_to=args.write_to)


def run(args):
    creator = DatasetCreator(dataset=args.data_path,
                             words_path=args.words_path,
                             pad_value=2,
                             word_height=args.word_height,
                             min_letter_size=37,
                             max_letter_size=41,
                             tall_to_low_letter_coef=1.3)

    labels, images = creator.create(words_count=args.words_count)
    np.save(os.path.join(args.save_to, 'labels.npy'), labels)
    np.save(os.path.join(args.save_to, 'images.npy'), images)


def run_labels_check(args):
    creator = DatasetCreator(dataset=args.data_path,
                             words_path=args.words_path,
                             pad_value=2,
                             word_height=args.word_height,
                             min_letter_size=37,
                             max_letter_size=41,
                             tall_to_low_letter_coef=1.3)
    labels = np.load(os.path.join(args.save_to, 'labels.npy'))
    images = np.load(os.path.join(args.save_to, 'images.npy'))
    for i in range(200, 250):
        image = images[i]
        print(creator.decode(label=labels[i]))
        image = image.reshape(64, 512)
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(r'C:\Users\heorhii.berezovskyi\Documents\words', str(i) + '.png'), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creates words dataset based on letter images.')
    parser.add_argument('--data_path', type=str, help='Path to letter images in a .npy file.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset')

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

    parser.add_argument('--word_height', type=int, help='Height of a result word image.', default=64)

    _args = parser.parse_args()

    # run(_args)
    run_labels_check(_args)
