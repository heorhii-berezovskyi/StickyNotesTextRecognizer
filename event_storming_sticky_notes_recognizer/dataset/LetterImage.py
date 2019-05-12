import argparse
import os

import cv2
import numpy as np
from numpy import ndarray

from event_storming_sticky_notes_recognizer.dataset.FileUtils import get_list_of_files


class LetterImage:
    def __init__(self, image: ndarray):
        self.letter = image

    def to_binary(self, thresh_value: int, dirty_frame_size: int):
        thresh, img_bin = cv2.threshold(self.letter, thresh_value, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bin = 255 - img_bin

        img_bin[:dirty_frame_size, :] = 0
        img_bin[-dirty_frame_size:, :] = 0
        img_bin[:, :dirty_frame_size] = 0
        img_bin[:, -dirty_frame_size:] = 0
        return LetterImage(image=img_bin)

    def save(self, to: str):
        if not os.path.exists(to):
            os.makedirs(to, exist_ok=True)
        cv2.imwrite(to, self.letter)

    def extract_roi(self, min_piece_area: int):
        letter = self.letter
        im2, contours, hierarchy = cv2.findContours(letter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_x = min_y = max(letter.shape)
        max_x = max_y = 0
        for c in contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)
            if w * h > min_piece_area:
                min_x = min(min_x, x)
                max_x = max(max_x, x + w)

                min_y = min(min_y, y)
                max_y = max(max_y, y + h)
        return LetterImage(image=letter[min_y:max_y, min_x:max_x])

    def with_morph_closing(self, kernel_size: int):
        letter = self.letter
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closing = cv2.morphologyEx(letter, cv2.MORPH_CLOSE, kernel)
        return LetterImage(image=closing)


def apply(args, dir: str):
    subfolders = os.listdir(os.path.join(args.directory, dir))
    for folder in subfolders:
        folder_path = os.path.join(args.directory, dir, folder)
        image_paths = get_list_of_files(folder=folder_path)

        for path in image_paths:
            letter = LetterImage(image=cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            binary_letter = letter.to_binary(thresh_value=args.thresh,
                                             dirty_frame_size=args.frame_size)
            closed_letter = binary_letter.with_morph_closing(kernel_size=args.kernel_size)
            roi = closed_letter.extract_roi(min_piece_area=args.min_area)
            roi.save(to=path)


def run(args):
    subdirs = os.listdir(args.directory)

    for dir in subdirs:
        if dir == 'marker' or dir == 'pen':
            apply(args=args, dir=dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts letters images into EMNIST format.')
    parser.add_argument('--directory', type=str,
                        help='Directory with marker and pen folders containing letters images to convert.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset')

    parser.add_argument('--thresh', type=int, help='Thresh value used to convert image into binary form.',
                        default=180)

    parser.add_argument('--frame_size', type=int, help='Letter frame size to clear.',
                        default=9)

    parser.add_argument('--kernel_size', type=int, help='Size of a kernel to perform closing morph op.',
                        default=5)

    parser.add_argument('--min_area', type=int, help='Minimal acceptable area of a roi piece.',
                        default=125)
    _args = parser.parse_args()
    run(_args)
