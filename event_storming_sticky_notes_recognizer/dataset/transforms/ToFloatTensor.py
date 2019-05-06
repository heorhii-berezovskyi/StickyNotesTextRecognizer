import torch

from event_storming_sticky_notes_recognizer.Name import Name


class ToFloatTensor:
    """Convert numpy arrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample[Name.IMAGE.value], sample[Name.LABEL.value]
        return {Name.IMAGE.value: torch.tensor(image, dtype=torch.float32),
                Name.LABEL.value: torch.tensor(label, dtype=torch.long)}
