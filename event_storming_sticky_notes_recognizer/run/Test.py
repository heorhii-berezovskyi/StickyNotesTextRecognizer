import cv2
import numpy as np
import torch
from torch.autograd import Variable
import argparse

from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder
from event_storming_sticky_notes_recognizer.run.models.crnn import CRNN


def run(args):
    data = np.load(args.data_path)

    model = CRNN(image_height=args.image_height,
                 num_of_channels=args.num_of_channels,
                 num_of_classes=args.num_of_classes,
                 num_of_lstm_hidden_units=args.num_of_lstm_hidden_units)

    print('loading pretrained model from %s' % args.model_path)
    model.load_state_dict(torch.load(args.model_path))

    converter = LabelEncoderDecoder()

    image = data[args.image_index]
    cv2.imshow('img', image)
    cv2.waitKey(0)

    image = image.reshape(1, args.num_of_channels, args.image_height, args.image_width)

    image = Variable(torch.FloatTensor(image))

    model.eval()
    preds = model(image)
    _, preds = preds.max(2)

    preds = preds.transpose(1, 0).contiguous().view(-1)

    print(preds)
    print(converter.decode_word(array=preds))
    print(converter.decode_word(array=converter.from_raw_to_label(array=preds)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs prediction for a given input image.')
    parser.add_argument('--model_path', type=str,
                        default=r'C:\Users\heorhii.berezovskyi\Documents\words\crnn151.pt',
                        help='Path to a model weights in .pt file')
    parser.add_argument('--data_path', type=str,
                        default=r'C:\Users\heorhii.berezovskyi\Documents\words\train_data.npy',
                        help='Path to an .npy file with images')

    parser.add_argument('--image_height', type=int, default=64,
                        help='Height of a single image to perform prediction on.')

    parser.add_argument('--image_width', type=int, default=512,
                        help='Width of a single image to perform prediction on.')

    parser.add_argument('--num_of_channels', type=int, default=1,
                        help='Number of symbols in alphabet including blank character.')

    parser.add_argument('--num_of_classes', type=int, default=27,
                        help='Number of channels in images.')

    parser.add_argument('--num_of_lstm_hidden_units', type=int, default=256,
                        help='Number of LSTM hidden units.')

    parser.add_argument('--image_index', type=int, default=0,
                        help='Index of an image in the .npy file to perform prediction on.')

    _args = parser.parse_args()
    run(args=_args)
