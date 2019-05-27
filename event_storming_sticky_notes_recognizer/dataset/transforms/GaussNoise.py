import cv2
import numpy as np

from event_storming_sticky_notes_recognizer.Name import Name


class GaussNoise:

    def __call__(self, sample):
        image, label, label_len = sample[Name.IMAGE.value], sample[Name.LABEL.value], sample[Name.LABEL_LEN.value]
        return {Name.IMAGE.value: self.noisy(image),
                Name.LABEL.value: label,
                Name.LABEL_LEN.value: label_len}

    def noisy(self, image, prob=1):
        apply_noise = np.random.binomial(1, prob)

        if apply_noise:
            image = image.transpose(1, 2, 0)
            image = image.astype(float)

            row, col, ch = image.shape

            mean = 0
            var = np.random.randint(1, 15)
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, (row, col))
            noisy_image = np.zeros(image.shape, np.float32)

            noisy_image[:, :, 0] = image[:, :, 0] + gaussian
            noisy_image[:, :, 1] = image[:, :, 1] + gaussian
            noisy_image[:, :, 2] = image[:, :, 2] + gaussian

            cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            noisy_image = noisy_image.astype(np.uint8)
            image = noisy_image.transpose(2, 0, 1)
        return image
