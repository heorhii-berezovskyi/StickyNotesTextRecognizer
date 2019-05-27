import cv2
import numpy as np

from event_storming_sticky_notes_recognizer.Name import Name


class Rotate:
    """Convert numpy arrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, label_len = sample[Name.IMAGE.value], sample[Name.LABEL.value], sample[Name.LABEL_LEN.value]
        return {Name.IMAGE.value: self.rotate(image),
                Name.LABEL.value: label,
                Name.LABEL_LEN.value: label_len}

    @staticmethod
    def rotate(img, rotate_prob=0.5, rotate_prec=250):
        rotate = np.random.binomial(1, rotate_prob)

        if rotate:
            img = img.transpose(1, 2, 0)
            img = img.astype(float)
            rows, cols, _ = img.shape
            rotate_prec = rotate_prec * max(rows / cols, cols / rows)
            rotate_angle = np.random.vonmises(0, kappa=rotate_prec) * 180 / np.pi
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_angle, 1)
            img = cv2.warpAffine(img, M, (cols, rows))
            img[img > 255.] = 255.
            img[img < 0.] = 0.
            img = img.astype(np.uint8)
            img = img.transpose(2, 0, 1)

        return img
