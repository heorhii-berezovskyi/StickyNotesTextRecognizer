import argparse
import os

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
    train_dataset = WordsDataset(data_set_dir=args.dataset_dir,
                                 data_set_type='train',
                                 transform=ToFloatTensor())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

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

    for epoch in range(1, args.epochs + 1):
        trainer.train(args=args, model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch)

        if args.save_model != '':
            torch.save(model.state_dict(),
                       os.path.join(args.save_model, 'crnn' + str(epoch) + '.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model with specified parameters.')

    parser.add_argument('--dataset_dir', type=str, default=r'D:\words\try',
                        help='Directory with dataset files train_labels.npy and train_data.npy.')

    parser.add_argument('--image_height', type=int, default=64, help='Height of input images.')
    parser.add_argument('--num_of_channels', type=int, default=1, help='Number of channels in input images.')
    parser.add_argument('--num_of_classes', type=int, default=27, help='Number of classes including blank character.')
    parser.add_argument('--num_of_lstm_hidden_units', type=int, default=256)

    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', default=r'D:\words\words',
                        help='Path to save the model')

    parser.add_argument('--loss', default=r'D:\words\words\loss',
                        help='Path to dump loss value')

    parser.add_argument('--pretrained', default='',
                        help='Path to a pretrained model weights.')
    _args = parser.parse_args()
    run(args=_args)

    # data = np.load(r'C:\Users\heorhii.berezovskyi\Documents\words\train_data.npy')
    # labels = np.load(r'C:\Users\heorhii.berezovskyi\Documents\words\train_labels.npy')
    #
    # save_data = data[:128]
    # save_labels = labels[:128]
    # np.save(r'C:\Users\heorhii.berezovskyi\Documents\words\try\train_data.npy', save_data)
    # np.save(r'C:\Users\heorhii.berezovskyi\Documents\words\try\train_labels.npy', save_labels)

    # data = np.load(r'C:\Users\heorhii.berezovskyi\Documents\words\try\train_data.npy')
    # labels = np.load(r'C:\Users\heorhii.berezovskyi\Documents\words\try\train_labels.npy')
    #
    # import cv2
    # i = 0
    # cv2.imshow('img', data[i])
    # print(data[i])
    # print(labels[i])
    # cv2.waitKey(0)
    # from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder
    #
    # print(LabelEncoderDecoder().decode_word(array=labels[i]))
