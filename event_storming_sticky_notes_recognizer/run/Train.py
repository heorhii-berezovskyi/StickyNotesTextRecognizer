import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader

from event_storming_sticky_notes_recognizer.dataset.EmnistDataset import EmnistDataset
from event_storming_sticky_notes_recognizer.dataset.transforms.ToFloatTensor import ToFloatTensor
from event_storming_sticky_notes_recognizer.model.Trainer import Trainer
from event_storming_sticky_notes_recognizer.run.models.ModelLoader import ModelLoader


def run(args):
    torch.manual_seed(args.seed)

    train_dataset = EmnistDataset(data_set_dir=r'C:\Users\heorhii.berezovskyi\Documents\emnist_balanced',
                                  data_set_type='train',
                                  transform=ToFloatTensor())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

    test_dataset = EmnistDataset(data_set_dir=r'C:\Users\heorhii.berezovskyi\Documents\emnist_balanced',
                                 data_set_type='test',
                                 transform=ToFloatTensor())
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=4)

    loader = ModelLoader()
    model = loader.load(name='simple_net')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    trainer = Trainer()

    for epoch in range(1, args.epochs + 1):
        trainer.train(args=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch)
        trainer.test(model=model, test_loader=test_loader)

        if args.save_model:
            torch.save(model.state_dict(),
                       os.path.join(r'C:\Users\heorhii.berezovskyi\Documents\emnist_balanced\models',
                                    'emnist_cnn' + str(epoch) + '.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model with specified parameters.')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='For Saving the current Model')
    _args = parser.parse_args()
    run(args=_args)
