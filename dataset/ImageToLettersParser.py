import os

import cv2


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


if __name__ == "__main__":
    # Read the image
    img = cv2.imread(r'C:\Users\heorhii.berezovskyi\Documents\letters\7.tif')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours for image, which will detect all the boxes
    im2, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    cropped_dir_path = r'C:\Users\heorhii.berezovskyi\Documents\LettersDataset\7'

    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        if 100 < w < 600 and 100 < h < 600:
            new_img = img[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(cropped_dir_path, (str(idx) + '.png')), new_img)
            # cv2.imwrite(cropped_dir_path + str(idx) + '.png', new_img)
            idx += 1
