import cv2
import numpy as np

from event_storming_sticky_notes_recognizer.Name import Name

BG_SIGMA = 5
MONOCHROME = 1


class Shear:
    def __call__(self, sample):
        image, label, label_len = sample[Name.IMAGE.value], sample[Name.LABEL.value], sample[Name.LABEL_LEN.value]
        return {Name.IMAGE.value: self.shear(image),
                Name.LABEL.value: label,
                Name.LABEL_LEN.value: label_len}

    @staticmethod
    def shear(img, shear_prob=0.5, shear_prec=150):
        shear = np.random.binomial(1, shear_prob)

        if shear:
            img = img.transpose(1, 2, 0)
            rows, cols, _ = img.shape
            shear_angle = np.random.vonmises(0, kappa=shear_prec)
            m = np.tan(shear_angle)

            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32([[50, 50], [200, 50], [50 + m * 150, 200]])
            M = cv2.getAffineTransform(pts1, pts2)

            img = cv2.warpAffine(img, M, (cols, rows))
            img = img.transpose(2, 0, 1)
        return img
