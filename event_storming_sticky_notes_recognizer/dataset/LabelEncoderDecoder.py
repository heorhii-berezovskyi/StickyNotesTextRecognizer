import numpy as np
from numpy import ndarray


class LabelEncoderDecoder:
    def __init__(self, max_word_len=16):
        self.length = max_word_len
        self.alphabet = ['a', 'b', 'c', 'd',
                         'e', 'f', 'g', 'h',
                         'i', 'j', 'k', 'l',
                         'm', 'n', 'o', 'p',
                         'q', 'r', 's', 't',
                         'u', 'v', 'w', 'x',
                         'y', 'z']

    def encode_character(self, character: str) -> int:
        assert len(character) == 1
        return self.alphabet.index(character) + 1

    def decode_character(self, value: int) -> str:
        assert value > 0
        return self.alphabet[value - 1]

    def encode_word(self, word: str) -> ndarray:
        encoded = np.zeros(self.length, dtype=np.uint8)
        for i in range(len(word)):
            encoded[i] = self.encode_character(character=word[i])
        return encoded

    def decode_word(self, array: ndarray) -> str:
        decoded = ''
        for element in array:
            if element == 0:
                decoded += '-'
            else:
                decoded += self.decode_character(value=element)
        return decoded

    def from_raw_to_label(self, array: ndarray) -> ndarray:
        word = []
        for i in range(len(array) - 1):
            if array[i + 1] != array[i] and array[i] != 0:
                word.append(array[i])
        word = np.asarray(word)
        result = np.zeros(self.length, dtype=np.uint8)
        if len(word) <= 16:
            result[:len(word)] = word
        else:
            result[:] = word[:16]
        return result

    @staticmethod
    def decode_word_len(array: ndarray) -> int:
        try:
            result = np.where(array == 0)[0][0]
            return result
        except Exception:
            return len(array)
