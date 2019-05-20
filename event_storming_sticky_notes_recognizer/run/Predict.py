import argparse

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder
from event_storming_sticky_notes_recognizer.run.models.crnn import CRNN


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def run(args):
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img', image)
    cv2.waitKey(0)

    thresh, img_bin = cv2.threshold(image, args.thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    cv2.imshow('img', img_bin)
    cv2.waitKey(0)

    img_bin = image_resize(image=img_bin, height=64)

    cv2.imshow('img', img_bin)
    cv2.waitKey(0)
    result = np.zeros((args.image_height, args.image_width))
    result[:, :img_bin.shape[1]] = img_bin

    cv2.imshow('img', result)
    cv2.waitKey(0)
    # data = np.load(args.data_path)

    model = CRNN(image_height=args.image_height,
                 num_of_channels=args.num_of_channels,
                 num_of_classes=args.num_of_classes,
                 num_of_lstm_hidden_units=args.num_of_lstm_hidden_units)

    print('loading pretrained model from %s' % args.model_path)
    model.load_state_dict(torch.load(args.model_path))

    converter = LabelEncoderDecoder()

    image = result

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
                        default=r'D:\words\models\crnn8.pt',
                        help='Path to a model weights in .pt file')
    parser.add_argument('--image_path', type=str,
                        default=r'D:\real_test_images\22.png',
                        help='Path to an image')

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

    parser.add_argument('--thresh', type=int, default=180,
                        help='Thresh value for binarization.')

    _args = parser.parse_args()
    run(args=_args)
