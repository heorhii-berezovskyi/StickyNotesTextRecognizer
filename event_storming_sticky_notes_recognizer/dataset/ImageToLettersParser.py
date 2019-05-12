import argparse
import os

import cv2
from numpy import ndarray

from event_storming_sticky_notes_recognizer.dataset.ContoursProcessor import ContoursProcessor
from event_storming_sticky_notes_recognizer.dataset.FileUtils import get_list_of_files, get_file_name
from event_storming_sticky_notes_recognizer.dataset.ImageUtils import save_images


class ImageToLettersParser:
    def __init__(self, min_letter_width: int, max_letter_width: int, min_letter_height: int, max_letter_height: int):
        self.min_w = min_letter_width
        self.max_w = max_letter_width
        self.min_h = min_letter_height
        self.max_h = max_letter_height

    def extract_letters(self, orig_image: ndarray, contours: list) -> list:
        letters = []
        for c in contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)
            if self.min_w < w < self.max_w and self.min_h < h < self.max_h:
                new_img = orig_image[y:y + h, x:x + w]
                letters.append(new_img)
        return letters


def apply(args, parser: ImageToLettersParser, dir_type: str):
    directory = os.path.join(args.directory, dir_type)
    image_paths = get_list_of_files(folder=directory)
    cont_processor = ContoursProcessor(thresh_value=args.thresh)
    path_to_save = os.path.join(args.write_to, dir_type)
    for path in image_paths:
        image_to_parse = cv2.imread(path)

        contours = cont_processor.find_contours(image=image_to_parse)

        sorted_contours, _ = cont_processor.sort_contours(contours=contours)

        letters = parser.extract_letters(orig_image=image_to_parse, contours=sorted_contours)

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save, exist_ok=True)
        save_images(letters=letters,
                    path_to=os.path.join(path_to_save,
                                         get_file_name(path=path)))


def run(args):
    parser = ImageToLettersParser(min_letter_width=args.min_letter_width,
                                  max_letter_width=args.max_letter_width,
                                  min_letter_height=args.min_letter_height,
                                  max_letter_height=args.max_letter_height)

    subdirs = os.listdir(args.directory)

    for dir in subdirs:
        if dir == 'marker' or dir == 'pen':
            apply(args=args, parser=parser, dir_type=dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts template with handwritten letters into letter images.')
    parser.add_argument('--min_letter_width', type=int, help='Minimal width of a letter on image.', default=400)
    parser.add_argument('--max_letter_width', type=int, help='Maximal width of a letter on image.', default=800)

    parser.add_argument('--min_letter_height', type=int, help='Minimal height of a letter on image.', default=400)
    parser.add_argument('--max_letter_height', type=int, help='Maximal height of a letter on image.', default=800)

    parser.add_argument('--thresh', type=int, help='Thresh value used to convert image into binary form.',
                        default=128)

    parser.add_argument('--directory', type=str,
                        help='directory with marker and pen folders containing template images with letters.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\letters')
    parser.add_argument('--write_to', type=str, help='Directory to save letters.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset')

    _args = parser.parse_args()
    run(_args)
