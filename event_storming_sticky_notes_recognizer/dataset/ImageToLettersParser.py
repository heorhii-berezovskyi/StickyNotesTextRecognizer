import argparse
import os

import cv2
from numpy import ndarray


class ImageToLettersParser:
    def __init__(self, min_letter_width: int, max_letter_width: int, min_letter_height: int, max_letter_height: int):
        self.min_w = min_letter_width
        self.max_w = max_letter_width
        self.min_h = min_letter_height
        self.max_h = max_letter_height

    def extract_letters(self, orig_image: ndarray, sorted_contours: list) -> list:
        letters = []
        for c in sorted_contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)
            if self.min_w < w < self.max_w and self.min_h < h < self.max_h:
                new_img = orig_image[y:y + h, x:x + w]
                letters.append(new_img)
        return letters

    @staticmethod
    def find_contours(image: ndarray, thresh_value: int) -> list:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh, img_bin = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY)

        cv2.imshow('img', img_bin)
        cv2.waitKey(0)

        # Find contours for image, which will detect all the boxes
        im2, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def save_letters(letters: list, path_to: str):
        if not os.path.exists(path_to):
            os.mkdir(path=path_to)
        for i in range(len(letters)):
            cv2.imwrite(os.path.join(path_to, (str(i) + '.png')), letters[i])

    @staticmethod
    def sort_contours(contours: list, method="left-to-right") -> (list, list):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes),
                                               key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return contours, bounding_boxes


def run(args):
    parser = ImageToLettersParser(min_letter_width=args.min_letter_width,
                                  max_letter_width=args.max_letter_width,
                                  min_letter_height=args.min_letter_height,
                                  max_letter_height=args.max_letter_height)

    image_to_parse = cv2.imread(args.image)

    contours = parser.find_contours(image=image_to_parse, thresh_value=args.thresh)

    sorted_contours, _ = parser.sort_contours(contours=contours,
                                              method="top-to-bottom")

    letters = parser.extract_letters(orig_image=image_to_parse,
                                     sorted_contours=sorted_contours)

    parser.save_letters(letters=letters,
                        path_to=args.write_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model with specified parameters.')
    parser.add_argument('--min_letter_width', type=int, help='Minimal width of a letter on image.', default=400)
    parser.add_argument('--max_letter_width', type=int, help='Maximal width of a letter on image.', default=800)

    parser.add_argument('--min_letter_height', type=int, help='Minimal height of a letter on image.', default=400)
    parser.add_argument('--max_letter_height', type=int, help='Maximal height of a letter on image.', default=800)

    parser.add_argument('--thresh', type=int, help='Thresh value used to convert image into binary form.',
                        default=240)

    parser.add_argument('--image', type=str, help='Image to parse path.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\letters\1.tif')
    parser.add_argument('--write_to', type=str, help='Directory to save letters.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset\1')

    _args = parser.parse_args()
    run(_args)
