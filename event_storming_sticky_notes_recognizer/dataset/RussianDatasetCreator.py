import argparse
import os

import cv2
import numpy as np
from numpy import ndarray

from event_storming_sticky_notes_recognizer.Exception import UnsupportedParamException
from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder


class EnglishDatasetCreator:
    def __init__(self, dataset: str, words_path: str, pad_value: int, word_height: int, min_letter_size: int,
                 max_letter_size: int, tall_to_low_letter_coef: float, label_encoder_decoder: LabelEncoderDecoder):
        self.directory = dataset
        self.word_list = self._load(file_path=words_path)
        self.pad_val = pad_value
        self.word_h = word_height
        self.min_size = min_letter_size
        self.max_size = max_letter_size
        self.coef = tall_to_low_letter_coef

        self.encoder_decoder = label_encoder_decoder

    def char_to_image(self, character: str):
        char_index = self.encoder_decoder.encode_character(character=character)
        random_image = np.random.randint(0, 472)
        image = cv2.imread(os.path.join(self.directory,
                                        str(char_index),
                                        (str(random_image) + '.png')
                                        ),
                           cv2.IMREAD_GRAYSCALE)
        if character in ['б', 'в']:
            return self.resize_long_letter(image, type='top')
        elif character in ['д', 'з', 'р', 'у', 'ф']:
            return self.resize_long_letter(image, type='bot')
        elif character in ['а', 'г', 'е', 'ж', 'и', 'й', 'к',
                           'л', 'м', 'н', 'о', 'п', 'с', 'т',
                           'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы',
                           'ь', 'э', 'ю', 'я']:
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
            letter_class = self.encoder_decoder.encode_character(character=word[i])
            letter_image = self.char_to_image(character=word[i])

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
        return np.asarray(labels, dtype=np.uint8), np.asarray(images, dtype=np.uint8)

    @staticmethod
    def _load(file_path: str) -> list:
        with open(file_path, encoding='Windows-1251') as myfile:
            head = [(next(myfile)).strip() for x in range(128)]
        print(head)
        return head


def run(args):
    label_encoder_decoder = LabelEncoderDecoder(max_word_len=args.max_length,
                                                alphabet=args.alphabet)
    creator = EnglishDatasetCreator(dataset=args.data_path,
                                    words_path=args.words_path,
                                    pad_value=1,
                                    word_height=args.word_height,
                                    min_letter_size=37,
                                    max_letter_size=41,
                                    tall_to_low_letter_coef=1.3,
                                    label_encoder_decoder=label_encoder_decoder)

    labels, images = creator.create(words_count=args.words_count)
    np.save(os.path.join(args.save_to, 'data.npy'), images)
    np.save(os.path.join(args.save_to, 'labels.npy'), labels)


def run_labels_check(args):
    label_encoder_decoder = LabelEncoderDecoder(max_word_len=args.max_length,
                                                alphabet=args.alphabet)
    labels = np.load(os.path.join(args.save_to, 'labels.npy'))
    images = np.load(os.path.join(args.save_to, 'data.npy'))
    for i in range(128):
        image = images[i]
        label = labels[i]
        print(label_encoder_decoder.decode_word(array=label))
        image = image.reshape(64, 512)
        cv2.imwrite(os.path.join(r'D:\russian_words\train\1', str(i) + '.png'), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates words dataset based on letter images.')

    parser.add_argument('--data_path', type=str,
                        help='Path to a directory containing marker and pen subdirectories with image folders.',
                        default=r'D:\russian_characters\experiment\small_roi')

    parser.add_argument('--words_path', type=str,
                        help='Path to a text file with words based on dataset will be created.',
                        default=r'D:\russian_words_corpus\russian.txt')

    parser.add_argument('--alphabet', type=str, help='Type of alphabet.', default='russian')

    parser.add_argument('--words_count', type=int, help='Number of word copies to create.', default=1)

    parser.add_argument('--save_to', type=str, help='Path to a directory to save created dataset.',
                        default=r'D:\russian_words\train\1')
    parser.add_argument('--max_length', type=int, help='Maximal number of letters in a single word.', default=16)
    parser.add_argument('--word_height', type=int, help='Height of a result word image.', default=64)

    _args = parser.parse_args()

    # run(_args)
    # run_labels_check(_args)


