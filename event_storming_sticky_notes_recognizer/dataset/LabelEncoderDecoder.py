import numpy as np
from numpy import ndarray


class LabelEncoderDecoder:
    def __init__(self, max_word_len: int):
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
                return decoded
            decoded += self.decode_character(value=element)
        return decoded
