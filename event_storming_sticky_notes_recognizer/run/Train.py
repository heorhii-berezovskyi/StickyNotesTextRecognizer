import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from event_storming_sticky_notes_recognizer.dataset.TestWordsDataset import TestWordsDataset
from event_storming_sticky_notes_recognizer.dataset.TrainWordsDataset import TrainWordsDataset
from event_storming_sticky_notes_recognizer.dataset.transforms.Erode import Erode
from event_storming_sticky_notes_recognizer.dataset.transforms.Rotate import Rotate
from event_storming_sticky_notes_recognizer.dataset.transforms.Shear import Shear
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

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.pretrained != '':
        print('loading pretrained model from %s' % args.pretrained)
        model.load_state_dict(torch.load(args.pretrained))
    else:
        model.apply(weights_init)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer()
    criterion = CTCLoss(reduction='mean')

    train_image = torch.FloatTensor(args.batch_size, 3, args.image_height, 512)
    test_image = torch.FloatTensor(args.test_batch_size, 3, args.image_height, 512)

    if args.cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(args.ngpu))
        train_image = train_image.cuda()
        test_image = test_image.cuda()
        criterion = criterion.cuda()

    train_image = Variable(train_image)
    test_image = Variable(test_image)

    for epoch in range(1, args.epochs + 1):
        test_losses = []
        if epoch % 5 == 0:
            for i in range(50):
                test_dataset = TestWordsDataset(data_set_dir=args.test_dataset_dir,
                                                transform=ToFloatTensor())

                test_loader = DataLoader(dataset=test_dataset,
                                         batch_size=args.test_batch_size,
                                         shuffle=False,
                                         num_workers=4)

                test_loss, test_accuracy = trainer.test(criterion=criterion,
                                                        model=model,
                                                        test_loader=test_loader,
                                                        test_image=test_image)
                test_losses.append(test_loss)
            test_losses_path = os.path.join(args.test_loss, 'losses.npy')
            try:
                losses_file = list(np.load(test_losses_path))
                np.save(test_losses_path, np.asarray(losses_file + [np.mean(test_losses)]))
            except FileNotFoundError:
                np.save(test_losses_path, np.asarray([np.mean(test_losses)]))
            print('Test Loss:', np.mean(test_losses))

        train_dataset = TrainWordsDataset(data_set_dir=args.train_dataset_dir,
                                          transform=transforms.Compose([Shear(),
                                                                        Erode(),
                                                                        Rotate(),
                                                                        ToFloatTensor()]))

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)
        losses = trainer.train(args=args,
                               criterion=criterion,
                               model=model,
                               train_loader=train_loader,
                               optimizer=optimizer,
                               epoch=epoch,
                               train_image=train_image)

        train_losses_path = os.path.join(args.train_loss, 'losses.npy')
        try:
            losses_file = list(np.load(train_losses_path))
            np.save(train_losses_path, np.asarray(losses_file + losses))
        except FileNotFoundError:
            np.save(train_losses_path, np.asarray(losses))

        if args.save_model != '':
            torch.save(model.state_dict(),
                       os.path.join(args.save_model, 'crnn' + str(epoch) + '.pt'))


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_losses(args):
    losses = np.load(os.path.join(args.train_loss), 'losses.npy')
    r_mean = running_mean(losses, 20)

    plt.ylim(0, 0.5)
    plt.plot(losses)
    plt.plot(r_mean)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model with specified parameters.')

    parser.add_argument('--train_dataset_dir', type=str, default=r'D:\russian_words\train',
                        help='Directory with folders containing data and labels in .npy format.')
    parser.add_argument('--test_dataset_dir', type=str, default=r'D:\russian_words\real\outputs',
                        help='Directory with folders containing data and labels in .npy format.')

    parser.add_argument('--image_height', type=int, default=64, help='Height of input images.')
    parser.add_argument('--num_of_channels', type=int, default=3, help='Number of channels in input images.')
    parser.add_argument('--num_of_classes', type=int, default=33, help='Number of classes including blank character.')
    parser.add_argument('--num_of_lstm_hidden_units', type=int, default=256)

    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=2, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')

    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', default=r'D:\russian_words\models',
                        help='Path to save the model')

    parser.add_argument('--train_loss', default=r'D:\russian_words\train_losses',
                        help='Path to dump train losses')

    parser.add_argument('--test_loss', default=r'D:\russian_words\test_losses',
                        help='Path to dump test losses')

    parser.add_argument('--test_acc', default=r'D:\russian_words\test_accuracies',
                        help='Path to dump test accuracies')

    parser.add_argument('--cuda', default=False,
                        help='Whether to enable training on gpu.')

    parser.add_argument('--pretrained', default=r'D:\russian_words\models\crnn0.pt',
                        help='Path to a pretrained model weights.')

    parser.add_argument('--ngpu', default=4, type=int)
    _args = parser.parse_args()
    run(args=_args)
