import argparse
import os
from random import shuffle

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from event_storming_sticky_notes_recognizer.dataset.WordsDataset import WordsDataset
from event_storming_sticky_notes_recognizer.dataset.transforms.ToFloatTensor import ToFloatTensor
from event_storming_sticky_notes_recognizer.model.Trainer import Trainer
from event_storming_sticky_notes_recognizer.run.models.crnn import CRNN


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, torch.nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def run(args):
    model = CRNN(image_height=args.image_height,
                 num_of_channels=args.num_of_channels,
                 num_of_classes=args.num_of_classes,
                 num_of_lstm_hidden_units=args.num_of_lstm_hidden_units)

    if args.pretrained != '':
        print('loading pretrained model from %s' % args.pretrained)
        model.load_state_dict(torch.load(args.pretrained))
    else:
        model.apply(weights_init)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer()

    train_dataset_folder_names = os.listdir(args.train_dataset_dir)
    # test_dataset_folder_names = os.listdir(args.test_dataset_dir)

    epoch_from = 1
    epoch_to = args.epochs + 9
    # test_losses = []
    # test_accuracies = []
    shuffle(train_dataset_folder_names)
    for train_folder_name in train_dataset_folder_names:
        train_path = os.path.join(args.train_dataset_dir, train_folder_name)

        train_dataset = WordsDataset(data_set_dir=train_path,
                                     transform=ToFloatTensor())

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)

        for epoch in range(epoch_from, epoch_to):
            losses = trainer.train(args=args,
                                   model=model,
                                   train_loader=train_loader,
                                   optimizer=optimizer,
                                   epoch=epoch)
            np.save(os.path.join(args.train_loss, 'train_losses' + str(epoch) + '.npy'), np.asarray(losses))

            if args.save_model != '':
                torch.save(model.state_dict(),
                           os.path.join(args.save_model, 'crnn' + str(epoch) + '.pt'))

        # for test_folder_name in test_dataset_folder_names:
        #     test_path = os.path.join(args.test_dataset_dir, test_folder_name)
        #     test_dataset = WordsDataset(data_set_dir=test_path,
        #                                 transform=ToFloatTensor())
        #
        #     test_loader = DataLoader(dataset=test_dataset,
        #                              batch_size=args.test_batch_size,
        #                              shuffle=False,
        #                              num_workers=4)
        #
        #     test_loss, test_accuracy = trainer.test(model=model,
        #                                             test_loader=test_loader)
        #     test_losses.append(test_loss)
        #     test_accuracies.append(test_accuracy)
        #     np.save(os.path.join(args.test_loss, 'test_losses' + str(epoch_to) + '.npy'),
        #             np.asarray(test_losses))
        #     np.save(os.path.join(args.test_acc, 'test_accuracies' + str(epoch_to) + '.npy'),
        #             np.asarray(test_accuracies))

        epoch_from += args.epochs
        epoch_to += args.epochs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model with specified parameters.')

    parser.add_argument('--train_dataset_dir', type=str, default=r'D:\russian_words\train',
                        help='Directory with folders containing data and labels in .npy format.')
    parser.add_argument('--test_dataset_dir', type=str, default=r'D:\words\test',
                        help='Directory with folders containing data and labels in .npy format.')

    parser.add_argument('--image_height', type=int, default=64, help='Height of input images.')
    parser.add_argument('--num_of_channels', type=int, default=1, help='Number of channels in input images.')
    parser.add_argument('--num_of_classes', type=int, default=33, help='Number of classes including blank character.')
    parser.add_argument('--num_of_lstm_hidden_units', type=int, default=256)

    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', default=r'D:\words\models',
                        help='Path to save the model')

    parser.add_argument('--train_loss', default=r'D:\words\train_losses',
                        help='Path to dump train losses')

    parser.add_argument('--test_loss', default=r'D:\words\test_losses',
                        help='Path to dump test losses')

    parser.add_argument('--test_acc', default=r'D:\words\test_accuracies',
                        help='Path to dump test accuracies')

    parser.add_argument('--pretrained', default='',
                        help='Path to a pretrained model weights.')
    _args = parser.parse_args()
    run(args=_args)

