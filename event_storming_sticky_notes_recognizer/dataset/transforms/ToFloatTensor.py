import torch

from event_storming_sticky_notes_recognizer.Name import Name


class ToFloatTensor:
    """Convert numpy arrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, label_len = sample[Name.IMAGE.value], sample[Name.LABEL.value], sample[Name.LABEL_LEN.value]
        # image = image.transpose(1, 2, 0)
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        # image = image.transpose(2, 0, 1)

        return {Name.IMAGE.value: torch.tensor(image, dtype=torch.float32),
                Name.LABEL.value: torch.tensor(label, dtype=torch.int32),
                Name.LABEL_LEN.value: torch.tensor(label_len, dtype=torch.int32)}
