import argparse
import os

import cv2
import numpy as np
from numpy import ndarray

from event_storming_sticky_notes_recognizer.Exception import UnsupportedParamException
from event_storming_sticky_notes_recognizer.dataset.ImageUtils import image_resize
from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder


class RussianDatasetCreator:
    def __init__(self, dataset: str, words_path: str, pad_value: int, word_height: int, min_letter_size: int,
                 max_letter_size: int, tall_to_low_letter_coef: float, label_encoder_decoder: LabelEncoderDecoder,
                 save_path: str):
        self.directory = dataset
        self.word_list = self._load(file_path=words_path)
        self.pad_val = pad_value
        self.word_h = word_height
        self.min_size = min_letter_size
        self.max_size = max_letter_size
        self.coef = tall_to_low_letter_coef

        self.encoder_decoder = label_encoder_decoder
        self.save_to = save_path

    def char_to_image(self, character: str, author: str):
        char_index = self.encoder_decoder.encode_character(character=character)
        random_image = np.random.randint(11)
        image = cv2.imread(os.path.join(self.directory,
                                        str(char_index),
                                        author,
                                        (str(random_image) + '.png')),
                           cv2.IMREAD_COLOR)
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
        h, w, c = letter.shape
        new_h = np.random.randint(self.min_size, self.max_size)
        resized = image_resize(image=letter, height=new_h)
        result = np.ones((self.word_h, resized.shape[1], c), dtype=np.uint8) * 255
        pad = np.random.randint(5, 10)
        if type == 'top':
            result[pad:resized.shape[0] + pad, :, :] = resized
        elif type == 'bot':
            result[self.word_h - resized.shape[0] - pad: self.word_h - pad, :, :] = resized
        else:
            raise UnsupportedParamException('Letter type ' + type + ' is not supported.')
        return result

    def resize_short_letter(self, letter: ndarray) -> ndarray:
        h, w, c = letter.shape
        new_h = np.random.randint(int(self.min_size / self.coef), int(self.max_size / self.coef))
        resized = image_resize(image=letter, height=new_h)
        result = np.ones((self.word_h, resized.shape[1], c), dtype=np.uint8) * 255

        third_random = np.random.randint(int(self.word_h / 3) - 2, int(self.word_h / 3) + 2)
        result[third_random: third_random + resized.shape[0], :, :] = resized
        return result

    def to_russian_image(self, word: str) -> (ndarray, ndarray):
        word = word.strip()
        word_label = np.zeros(16, dtype=np.uint16)
        word_image = np.ones((64, 1, 3), dtype=np.uint8) * 255
        author_id = str(np.random.randint(43))
        for i in range(len(word)):
            letter_class = self.encoder_decoder.encode_character(character=word[i])
            letter_image = self.char_to_image(character=word[i], author=author_id)
            word_image = np.hstack((word_image, letter_image))
            word_label[i] = letter_class
        return word_label, word_image

    def create_russian(self):
        width_offset = 35
        page_height = 2200
        page_width = 1800
        occupied_h = 0
        occupied_w = width_offset
        page = np.ones((page_height, page_width, 3), dtype=np.uint8) * 255
        labels = []
        i = 0
        for word in self.word_list:
            word_label, word_image = self.to_russian_image(word=word)
            word_w = word_image.shape[1]
            if occupied_w + word_w <= page_width:

                min_h = occupied_h
                max_h = occupied_h + self.word_h
                min_w = occupied_w
                max_w = occupied_w + word_w

                page[min_h: max_h, min_w: max_w, :] = word_image
                coords = np.array([min_h, max_h, min_w, max_w], dtype=np.uint16)
                labels.append(np.hstack((word_label, coords)))
                occupied_w += word_w
                if occupied_w + width_offset <= page_width:
                    occupied_w += width_offset
            elif occupied_h + 2 * self.word_h <= page_height:
                occupied_w = width_offset
                occupied_h += self.word_h

                min_h = occupied_h
                max_h = occupied_h + self.word_h
                min_w = occupied_w
                max_w = occupied_w + word_w

                page[min_h: max_h, min_w: max_w, :] = word_image
                coords = np.array([min_h, max_h, min_w, max_w], dtype=np.uint16)
                labels.append(np.hstack((word_label, coords)))

                occupied_w += word_w
                if occupied_w + width_offset <= page_width:
                    occupied_w += width_offset
            else:
                save_folder = os.path.join(self.save_to, str(i))
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=True)

                cv2.imwrite(os.path.join(save_folder, 'page.png'), page)
                i += 1
                np.save(os.path.join(save_folder, 'labels.npy'), np.asarray(labels))
                labels = []
                page = np.ones((page_height, page_width, 3), dtype=np.uint8) * 255
                occupied_h = 0
                occupied_w = width_offset

    @staticmethod
    def _load(file_path: str) -> list:
        with open(file_path, encoding='utf-8') as myfile:
            head = [x.strip() for x in myfile]
        return head


def run(args):
    label_encoder_decoder = LabelEncoderDecoder(max_word_len=args.max_length,
                                                alphabet=args.alphabet)
    creator = RussianDatasetCreator(dataset=args.data_path,
                                    words_path=args.words_path,
                                    pad_value=0,
                                    word_height=args.word_height,
                                    min_letter_size=37,
                                    max_letter_size=41,
                                    tall_to_low_letter_coef=1.5,
                                    label_encoder_decoder=label_encoder_decoder,
                                    save_path=args.save_to)

    creator.create_russian()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates words dataset based on letter images.')

    parser.add_argument('--data_path', type=str,
                        help='Path to a directory containing subdirectories with folders from 1 to 32(russian letters) each containing subfolders of letters written by a unique author.',
                        default=r'D:\russian_characters\experiment\small_roi')

    parser.add_argument('--words_path', type=str,
                        help='Path to a text file with words based on dataset will be created.',
                        default=r'D:\russian_words_corpus\russian_unicode_filtered.txt')

    parser.add_argument('--alphabet', type=str, help='Type of alphabet.', default='russian')

    parser.add_argument('--words_count', type=int, help='Number of word copies to create.', default=1)

    parser.add_argument('--save_to', type=str, help='Path to a directory to save created dataset.',
                        default=r'D:\russian_words\train')
    parser.add_argument('--max_length', type=int, help='Maximal number of letters in a single word.', default=16)
    parser.add_argument('--word_height', type=int, help='Height of a result word image.', default=64)

    _args = parser.parse_args()

    run(_args)


