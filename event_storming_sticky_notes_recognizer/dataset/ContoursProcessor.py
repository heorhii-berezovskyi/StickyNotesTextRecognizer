import cv2
from numpy import ndarray


class ContoursProcessor:
    def __init__(self, thresh_value: int):
        self.thresh = thresh_value

    def find_contours(self, image: ndarray) -> list:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh, img_bin = cv2.threshold(img_gray, self.thresh, 255, cv2.THRESH_BINARY)

        # Find contours for image, which will detect all the boxes
        im2, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def sort_contours(contours: list) -> (list, list):
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                                key=lambda b: b[1][1], reverse=False))

        # return the list of sorted contours and bounding boxes
        return contours, boundingBoxes
