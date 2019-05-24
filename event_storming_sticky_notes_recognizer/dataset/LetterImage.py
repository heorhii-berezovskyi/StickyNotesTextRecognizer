import argparse
import os

import cv2
import numpy as np
from numpy import ndarray

from event_storming_sticky_notes_recognizer.dataset.FileUtils import get_list_of_files
from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder


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

    def save(self, to: str, name: str):
        if not os.path.exists(to):
            os.makedirs(to, exist_ok=True)
        path = os.path.join(to, name)
        cv2.imwrite(path, self.letter)

    def extract_roi(self, orig_image: ndarray, min_piece_area: int):
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
        return LetterImage(image=orig_image[min_y:max_y, min_x:max_x, :])

    def with_morph_closing(self, kernel_size: int):
        letter = self.letter
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closing = cv2.morphologyEx(letter, cv2.MORPH_CLOSE, kernel)
        return LetterImage(image=closing)

    def deskew(self):
        img = self.letter
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            # no deskewing needed.
            return LetterImage(image=img.copy())
        # Calculate skew based on central momemts.
        skew = m['mu11'] / m['mu02']
        # Calculate affine transform to correct skewness.
        M = np.float32([[1, skew, -0.5 * img.shape[0] * skew], [0, 1, 0]])
        # Apply affine transform
        img = cv2.warpAffine(img, M, img.shape, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return LetterImage(image=img)


def findnth(haystack, needle, n):
    parts = haystack.split(needle, n + 1)
    if len(parts) <= n + 1:
        return -1
    return len(haystack) - len(parts[-1]) - len(needle)


def run(args):
    subdirs = os.listdir(args.directory)

    for dir in subdirs:
        subfolders = os.listdir(os.path.join(args.directory, dir))
        for folder in subfolders:
            folder_path = os.path.join(args.directory, dir, folder)
            image_paths = get_list_of_files(folder=folder_path)

            for path in image_paths:
                letter = LetterImage(image=cv2.imread(path, cv2.IMREAD_GRAYSCALE))
                binary_letter = letter.to_binary(thresh_value=args.thresh,
                                                 dirty_frame_size=args.frame_size)
                closed_letter = binary_letter.with_morph_closing(kernel_size=args.kernel_size)
                # roi = closed_letter.extract_roi(min_piece_area=args.min_area)
                closed_letter.save(to=path, name='')


def run_russian(args):
    encoder_decoder = LabelEncoderDecoder(alphabet='russian')
    subdirs = os.listdir(args.directory)

    for dir in subdirs:
        dir_path = os.path.join(args.directory, dir)

        image_names = os.listdir(dir_path)
        for name in image_names:
            path = os.path.join(dir_path, name)
            stream = open(path, 'rb')
            bytes = bytearray(stream.read())
            array_path = np.asarray(bytes, dtype=np.uint8)
            image = cv2.imdecode(array_path, cv2.IMREAD_COLOR)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            letter = LetterImage(image=gray_image)
            binary_letter = letter.to_binary(thresh_value=args.thresh,
                                             dirty_frame_size=args.frame_size)
            closed_letter = binary_letter.with_morph_closing(kernel_size=args.kernel_size)
            # de_skewed_letter = closed_letter.deskew()
            roi = closed_letter.extract_roi(orig_image=image,
                                            min_piece_area=args.min_area)
            try:
                author_id = int(name[:4])
                save_path = os.path.join(args.save_dir,
                                         str(encoder_decoder.encode_character(character=dir)),
                                         str(author_id))
                save_name = int(name[6 + 1: -4])
                roi.save(to=save_path, name=str(save_name) + '.png')
            except:
                print(dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts letters images into EMNIST format.')
    parser.add_argument('--directory', type=str,
                        help='Directory with marker and pen folders containing letters images to convert.',
                        default=r'D:\russian_characters\experiment\small')

    parser.add_argument('--save_dir', type=str,
                        help='Directory to save russian letters.',
                        default=r'D:\russian_characters\experiment\small_roi')

    parser.add_argument('--thresh', type=int, help='Thresh value used to convert image into binary form.',
                        default=128)

    parser.add_argument('--frame_size', type=int, help='Letter frame size to clear.',
                        default=9)

    parser.add_argument('--kernel_size', type=int, help='Size of a kernel to perform closing morph op.',
                        default=5)

    parser.add_argument('--min_area', type=int, help='Minimal acceptable area of a roi piece.',
                        default=125)
    _args = parser.parse_args()
    # run(_args)
    run_russian(_args)
