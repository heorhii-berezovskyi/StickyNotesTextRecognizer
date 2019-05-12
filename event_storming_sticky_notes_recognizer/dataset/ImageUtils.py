import os

import cv2


def save_images(letters: list, path_to: str):
    if not os.path.exists(path_to):
        os.makedirs(path_to, exist_ok=True)
    for i in range(len(letters)):
        cv2.imwrite(os.path.join(path_to, (str(i) + '.png')), letters[i])
