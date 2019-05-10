import argparse
import glob
import os

import cv2
import numpy as np


class ToEmnistDataConverter:

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
    def center_frame(images: list) -> list:
        squares = []
        for image in images:
            height, width = image.shape
            if height > width:
                differ = height
            else:
                differ = width

            mask = np.zeros((differ, differ), dtype=np.uint8)
            x_pos = int((differ - width) / 2)
            y_pos = int((differ - height) / 2)
            mask[y_pos:y_pos + height, x_pos:x_pos + width] = image[0:height, 0:width]
            # thresh, img_bin = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY)
            squares.append(mask)
        return squares

    @staticmethod
    def resize_and_resample(images: list, size: int) -> list:
        results = []
        for image in images:
            resized = cv2.resize(image, (size - 4, size - 4), interpolation=cv2.INTER_AREA)
            padded = np.pad(resized, ((2, 2), (2, 2)), 'constant')
            padded = padded * 1.7
            padded[padded > 255.] = 255.
            padded = padded.astype(np.uint8)
            results.append(padded)
        return results

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
    converter = ToEmnistDataConverter()
    image_names = glob.glob(args.img_dir)
    images = converter.read_images(image_names=image_names)
    binary_images = converter.to_binary(images=images,
                                        thresh_value=args.thresh)

    converter.save_letters(letters=binary_images,
                           letters_names=image_names,
                           path_to=args.write_to)


def run_extract_roi(args):
    converter = ToEmnistDataConverter()
    image_names = glob.glob(args.img_dir)
    images = converter.read_images(image_names=image_names)

    rois = converter.extract_roi(images=images)

    converter.save_letters(letters=rois,
                           letters_names=image_names,
                           path_to=args.write_to)


def run_center_frame(args):
    converter = ToEmnistDataConverter()
    image_names = glob.glob(args.img_dir)
    images = converter.read_images(image_names=image_names)

    squares = converter.center_frame(images=images)

    converter.save_letters(letters=squares,
                           letters_names=image_names,
                           path_to=args.write_to)


def run_resize_resample(args):
    converter = ToEmnistDataConverter()
    image_names = glob.glob(args.img_dir)
    images = converter.read_images(image_names=image_names)

    resized_squares = converter.resize_and_resample(images=images,
                                                    size=args.final_size)
    converter.save_letters(letters=resized_squares,
                           letters_names=image_names,
                           path_to=args.write_to)


def run_add_closing(args):
    converter = ToEmnistDataConverter()
    image_names = glob.glob(args.img_dir)
    images = converter.read_images(image_names=image_names)

    closings = converter.add_closing(images=images)
    converter.save_letters(letters=closings,
                           letters_names=image_names,
                           path_to=args.write_to)


def run_all(args):
    converter = ToEmnistDataConverter()
    image_names = glob.glob(args.img_dir)
    images = converter.read_images(image_names=image_names)

    binary_images = converter.to_binary(images=images,
                                        thresh_value=args.thresh)

    closings = converter.add_closing(images=binary_images)

    rois = converter.extract_roi(images=closings)

    centered = converter.center_frame(images=rois)

    resized = converter.resize_and_resample(images=centered,
                                            size=args.final_size)

    converter.save_letters(letters=resized,
                           letters_names=image_names,
                           path_to=args.write_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts letters images into EMNIST format.')
    parser.add_argument('--img_dir', type=str, help='Directory with images to process.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset\10\*')

    parser.add_argument('--thresh', type=int, help='Thresh value used to convert image into binary form.',
                        default=180)

    parser.add_argument('--write_to', type=str, help='Directory to save letters.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset\10')

    parser.add_argument('--final_size', type=int, help='Final image size accordingly to a emnist dataset or others.',
                        default=28)

    _args = parser.parse_args()

    # run_to_binary(_args)
    # run_add_closing(_args)
    # run_extract_roi(_args)
    # run_center_frame(_args)
    # run_resize_resample(_args)
    run_all(_args)
