import cv2
import numpy as np

from event_storming_sticky_notes_recognizer.Name import Name


class Erode:
    """Convert numpy arrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, label_len = sample[Name.IMAGE.value], sample[Name.LABEL.value], sample[Name.LABEL_LEN.value]
        return {Name.IMAGE.value: self.erode(image),
                Name.LABEL.value: label,
                Name.LABEL_LEN.value: label_len}

    @staticmethod
    def erode(img, erosion_prob=0.5, erosion_srate=1, erosion_rrate=1.2):

        erode = np.random.binomial(1, erosion_prob)

        if erode:
            img = img.transpose(1, 2, 0)
            kernel_size = np.min([2 * np.random.geometric(erosion_srate) + 1, 15])
            kernel = np.zeros([kernel_size, kernel_size])
            center = np.array([int(kernel_size / 2), int(kernel_size / 2)])
            for x in range(kernel_size):
                for y in range(kernel_size):
                    d = np.linalg.norm(np.array([x, y]) - center)
                    p = np.exp(-d * 1)
                    value = np.random.binomial(1, p)
                    kernel[x, y] = value or 10 ** -16

            img = cv2.erode(img, kernel, iterations=1)
            img = img.transpose(2, 0, 1)

        return img
