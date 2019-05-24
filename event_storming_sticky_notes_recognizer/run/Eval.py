import argparse

import torch
from torch.utils.data import DataLoader

from event_storming_sticky_notes_recognizer.dataset.TrainWordsDataset import TrainWordsDataset
from event_storming_sticky_notes_recognizer.dataset.transforms.ToFloatTensor import ToFloatTensor
from event_storming_sticky_notes_recognizer.model.Trainer import Trainer
from event_storming_sticky_notes_recognizer.run.models.crnn import CRNN


def run(args):
    test_dataset = TrainWordsDataset(data_set_dir=args.dataset_dir,
                                     transform=ToFloatTensor())

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=4)

    model = CRNN(image_height=args.image_height,
                 num_of_channels=args.num_of_channels,
                 num_of_classes=args.num_of_classes,
                 num_of_lstm_hidden_units=args.num_of_lstm_hidden_units)

    model.load_state_dict(torch.load(args.snapshot))
    print(model)

    trainer = Trainer()

    trainer.test(model=model, test_loader=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model with specified parameters.')

    parser.add_argument('--dataset_dir', type=str, default=r'C:\Users\heorhii.berezovskyi\Documents\words',
                        help='Directory with dataset files train_labels.npy and train_data.npy.')

    parser.add_argument('--image_height', type=int, default=64, help='Height of input images.')
    parser.add_argument('--num_of_channels', type=int, default=1, help='Number of channels in input images.')
    parser.add_argument('--num_of_classes', type=int, default=27, help='Number of classes including blank character.')
    parser.add_argument('--num_of_lstm_hidden_units', type=int, default=256)

    parser.add_argument('--test_batch_size', type=int, default=512, metavar='N',
                        help='input batch size for testing')

    parser.add_argument('--snapshot', default=r'C:\Users\heorhii.berezovskyi\Documents\words\crnn151.pt',
                        help='Path to a pretrained model weights.')
    _args = parser.parse_args()
    run(args=_args)
