import os

import cv2


def save_images(letters: list, path_to: str):
    if not os.path.exists(path_to):
        os.makedirs(path_to, exist_ok=True)
    for i in range(len(letters)):
        cv2.imwrite(os.path.join(path_to, (str(i) + '.png')), letters[i])


def image_resize(image, width=None, height=None):
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
        if height >= h:
            print(True)
            inter = cv2.INTER_CUBIC
        else:
            inter = cv2.INTER_AREA

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
        if width >= w:
            print(True)
            inter = cv2.INTER_CUBIC
        else:
            inter = cv2.INTER_AREA

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
