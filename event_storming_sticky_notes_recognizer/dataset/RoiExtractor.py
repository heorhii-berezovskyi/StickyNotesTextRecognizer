import argparse
import glob
import os

import cv2
import numpy as np

from event_storming_sticky_notes_recognizer.Exception import UnsupportedParamException


class RoiExtractor:

    @staticmethod
    def to_binary(images: list, thresh_value: int) -> list:
        binary_images = []
        for image in images:
            thresh, img_bin = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img_bin = 255 - img_bin

            img_bin[:9, :] = 0
            img_bin[-9:, :] = 0
            img_bin[:, :9] = 0
            img_bin[:, -9:] = 0

            binary_images.append(img_bin)
        return binary_images

    @staticmethod
    def save_letters(letters: list, letters_names: list, path_to: str):
        if not os.path.exists(path_to):
            os.mkdir(path=path_to)
        for i in range(len(letters)):
            head, tail = os.path.split(letters_names[i])
            cv2.imwrite(os.path.join(path_to, tail), letters[i])

    @staticmethod
    def extract_roi(images: list) -> list:
        extracted_rois = []
        for image in images:
            im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            min_x = min_y = 10000
            max_x = max_y = 0
            for c in contours:
                # Returns the location and width,height for every contour
                x, y, w, h = cv2.boundingRect(c)
                if w * h > 125:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x + w)

                    min_y = min(min_y, y)
                    max_y = max(max_y, y + h)
            extracted_rois.append(image[min_y:max_y, min_x:max_x])
        return extracted_rois

    @staticmethod
    def add_closing(images: list) -> list:
        results = []
        kernel = np.ones((5, 5), np.uint8)
        for image in images:
            closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            results.append(closing)
        return results

    @staticmethod
    def read_images(image_names: list):
        images = []
        for name in image_names:
            image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
            images.append(image)
        return images


def run_to_binary(args):
    converter = RoiExtractor()
    image_names = glob.glob(args.img_dir)
    images = converter.read_images(image_names=image_names)
    binary_images = converter.to_binary(images=images,
                                        thresh_value=args.thresh)

    converter.save_letters(letters=binary_images,
                           letters_names=image_names,
                           path_to=args.write_to)


def run_extract_roi(args):
    converter = RoiExtractor()
    image_names = glob.glob(args.img_dir)
    images = converter.read_images(image_names=image_names)

    rois = converter.extract_roi(images=images)

    converter.save_letters(letters=rois,
                           letters_names=image_names,
                           path_to=args.write_to)


def run_add_closing(args):
    converter = RoiExtractor()
    image_names = glob.glob(args.img_dir)
    images = converter.read_images(image_names=image_names)

    closings = converter.add_closing(images=images)
    converter.save_letters(letters=closings,
                           letters_names=image_names,
                           path_to=args.write_to)


def apply(args, converter: RoiExtractor, dir: str, type: str):
    subfolders = os.listdir(os.path.join(args.directory, dir))
    for folder in subfolders:
        image_paths = glob.glob(os.path.join(args.directory, dir, folder) + r'\*')
        images = converter.read_images(image_names=image_paths)
        binary_images = converter.to_binary(images=images, thresh_value=args.thresh)
        closings = converter.add_closing(images=binary_images)
        rois = converter.extract_roi(images=closings)
        converter.save_letters(letters=rois,
                               letters_names=image_paths,
                               path_to=os.path.join(args.write_to, type, folder))


def run_all(args):
    converter = RoiExtractor()
    subdirs = os.listdir(args.directory)

    for dir in subdirs:
        if dir == 'marker' or dir == 'pen':
            apply(args=args, converter=converter, dir=dir, type=dir)
        else:
            raise UnsupportedParamException('Folder with name ' + dir + ' is not supported for handling.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts letters images into EMNIST format.')
    parser.add_argument('--directory', type=str,
                        help='Directory with marker and pen folders containing letters images to convert.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset')

    parser.add_argument('--thresh', type=int, help='Thresh value used to convert image into binary form.',
                        default=180)

    parser.add_argument('--write_to', type=str, help='Directory to save letters.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset')

    parser.add_argument('--final_size', type=int, help='Final image size accordingly to a emnist dataset or others.',
                        default=28)

    _args = parser.parse_args()

    # run_to_binary(_args)
    # run_add_closing(_args)
    # run_extract_roi(_args)
    run_all(_args)
