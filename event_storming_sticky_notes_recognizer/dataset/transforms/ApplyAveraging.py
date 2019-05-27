import cv2
import numpy as np
from event_storming_sticky_notes_recognizer.Name import Name


class ApplyAveraging:
    """Convert numpy arrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, label_len = sample[Name.IMAGE.value], sample[Name.LABEL.value], sample[Name.LABEL_LEN.value]
        return {Name.IMAGE.value: self.apply_averaging(image),
                Name.LABEL.value: label,
                Name.LABEL_LEN.value: label_len}

    @staticmethod
    def apply_averaging(img, prob=0.5):
        apply_prob = np.random.binomial(1, prob)

        if apply_prob:
            img = img.transpose(1, 2, 0)
            blur = cv2.blur(img, (5, 5))
            img = blur.transpose(2, 0, 1)
        return img
