import argparse
import glob
import os

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter


class ToEmnistDataConverter:

    @staticmethod
    def to_binary(image_names: list, thresh_value: int, thresh_mode: str) -> list:
        binary_images = []
        for name in image_names:
            image = cv2.imread(name)
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if thresh_mode == 'binary':
                thresh, img_bin = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY)
            else:
                thresh, img_bin = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            img_bin = 255 - img_bin
            binary_images.append(img_bin)
        return binary_images

    @staticmethod
    def add_gaussian_blur(image_names: list) -> list:
        blurred_images = []
        for name in image_names:
            image = cv2.imread(name)
            blurred = gaussian_filter(image, sigma=1)
            blurred_images.append(blurred)
        return blurred_images

    @staticmethod
    def save_letters(letters: list, letters_names: list, path_to: str):
        if not os.path.exists(path_to):
            os.mkdir(path=path_to)
        for i in range(len(letters)):
            head, tail = os.path.split(letters_names[i])
            cv2.imwrite(os.path.join(path_to, tail), letters[i])

    @staticmethod
    def extract_roi(image_names: list, num_dilate_iters: int) -> list:
        extracted_rois = []
        kernel = np.ones((5, 5), np.uint8)
        for name in image_names:
            image = cv2.imread(name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.dilate(gray, kernel=kernel, iterations=num_dilate_iters)
            im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            letters = []
            for c in contours:
                # Returns the location and width,height for every contour
                x, y, w, h = cv2.boundingRect(c)
                if 40 < w < 600 and 40 < h < 600:
                    new_img = image[y:y + h, x:x + w]
                    letters.append(new_img)
            extracted_rois.append(letters[0])
        return extracted_rois

    @staticmethod
    def center_frame(image_names: list) -> list:
        squares = []
        for name in image_names:
            image = cv2.imread(name)
            height, width, channels = image.shape
            if height > width:
                differ = height
            else:
                differ = width
            differ += 4

            mask = np.zeros((differ, differ, channels), dtype=np.uint8)
            x_pos = int((differ - width) / 2)
            y_pos = int((differ - height) / 2)
            mask[y_pos:y_pos + height, x_pos:x_pos + width] = image[0:height, 0:width]
            squares.append(mask)
        return squares

    @staticmethod
    def resize_and_resample(image_names: list, size: int) -> list:
        results = []
        for name in image_names:
            image = cv2.imread(name)
            resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
            results.append(resized)
        return results


def run_to_binary(args):
    converter = ToEmnistDataConverter()
    image_names = glob.glob(args.img_dir)
    binary_images = converter.to_binary(image_names=image_names,
                                        thresh_value=args.thresh,
                                        thresh_mode=args.thresh_mode)

    converter.save_letters(letters=binary_images,
                           letters_names=image_names,
                           path_to=args.write_to)


def run_add_gaussian_blur(args):
    converter = ToEmnistDataConverter()
    image_names = glob.glob(args.img_dir)
    blurred_images = converter.add_gaussian_blur(image_names=image_names)

    converter.save_letters(letters=blurred_images,
                           letters_names=image_names,
                           path_to=args.write_to)


def run_extract_roi(args):
    converter = ToEmnistDataConverter()
    image_names = glob.glob(args.img_dir)

    if not args.by_index:
        rois = converter.extract_roi(image_names=image_names,
                                     num_dilate_iters=args.num_dilate_iters)

        converter.save_letters(letters=rois,
                               letters_names=image_names,
                               path_to=args.write_to)
    else:
        image_name = [args.image_path]
        roi = converter.extract_roi(image_names=image_name,
                                    num_dilate_iters=args.num_dilate_iters)

        converter.save_letters(letters=roi,
                               letters_names=image_name,
                               path_to=args.write_to)


def run_center_frame(args):
    converter = ToEmnistDataConverter()
    image_names = glob.glob(args.img_dir)

    squares = converter.center_frame(image_names=image_names)

    converter.save_letters(letters=squares,
                           letters_names=image_names,
                           path_to=args.write_to)


def run_resize_resample(args):
    converter = ToEmnistDataConverter()
    image_names = glob.glob(args.img_dir)

    resized_squares = converter.resize_and_resample(image_names=image_names,
                                                    size=args.final_size)
    converter.save_letters(letters=resized_squares,
                           letters_names=image_names,
                           path_to=args.write_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts template with handwritten letters into letter images.')
    parser.add_argument('--img_dir', type=str, help='Directory with images to process.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset\12_bin_blurred\*')

    parser.add_argument('--thresh', type=int, help='Thresh value used to convert image into binary form.',
                        default=180)

    parser.add_argument('--write_to', type=str, help='Directory to save letters.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset\12_bin_blurred_roi')

    parser.add_argument('--final_size', type=int, help='Final image size accordingly to a emnist dataset or others.',
                        default=28)

    parser.add_argument('--thresh_mode', type=str, help='Thresh mode in opencv thresh holding. (binary, otsu)',
                        default='binary')

    parser.add_argument('--num_dilate_iters', type=int,
                        help='Number of iterations to perform dilation while extracting roi',
                        default=10)

    parser.add_argument('--by_index', type=bool, help='Whether to extract roi of a single image', default=True)
    parser.add_argument('--image_path', type=str, help='path to an image to extract roi from',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset\12_bin_blurred\19.png')

    _args = parser.parse_args()

    # run_to_binary(_args)
    # run_add_gaussian_blur(_args)
    run_extract_roi(_args)
    # run_center_frame(_args)
    # run_resize_resample(_args)
