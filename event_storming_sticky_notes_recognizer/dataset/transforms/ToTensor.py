import torch
from event_storming_sticky_notes_recognizer.Name import Name


class ToTensor:
    """Convert numpy arrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample[Name.IMAGE], sample[Name.LABEL]
        return {Name.IMAGE: torch.from_numpy(image),
                Name.LABEL: label}
