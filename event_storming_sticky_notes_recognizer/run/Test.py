import cv2
import numpy as np
import torch
from torch.autograd import Variable
import argparse
import json
import os

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


def run_test_real(args):
    model = CRNN(image_height=args.image_height,
                 num_of_channels=args.num_of_channels,
                 num_of_classes=args.num_of_classes,
                 num_of_lstm_hidden_units=args.num_of_lstm_hidden_units)

    print('loading pretrained model from %s' % args.model_path)
    model.load_state_dict(torch.load(args.model_path))

    converter = LabelEncoderDecoder(alphabet='russian')

    with open(args.data_path, encoding='utf-8') as f:
        data = json.load(f)

    page = cv2.imread(data['path'], cv2.IMREAD_COLOR)
    i = 6
    print(data['outputs']['object'])
    x_min = data['outputs']['object'][i]['bndbox']['xmin']
    y_min = data['outputs']['object'][i]['bndbox']['ymin']
    x_max = data['outputs']['object'][i]['bndbox']['xmax']
    y_max = data['outputs']['object'][i]['bndbox']['ymax']

    image = page[y_min: y_max, x_min: x_max, :]
    image = image_resize(image, height=54)

    image_height = image.shape[0]
    image_width = image.shape[1]

    result = np.ones((64, 512, 3), dtype=np.uint8) * 255

    result[5: image_height + 5, 0: image_width, :] = image
    image = result
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


def run_test_synthetic(args):
    model = CRNN(image_height=args.image_height,
                 num_of_channels=args.num_of_channels,
                 num_of_classes=args.num_of_classes,
                 num_of_lstm_hidden_units=args.num_of_lstm_hidden_units)

    print('loading pretrained model from %s' % args.model_path)
    model.load_state_dict(torch.load(args.model_path))

    folders = os.listdir(args.data_path)
    random_folder = np.random.randint(len(folders))
    path = os.path.join(args.data_path, str(random_folder))
    page = cv2.imread(os.path.join(path, 'page.png'), cv2.IMREAD_COLOR)
    label_data = np.load(os.path.join(path, 'labels.npy'))
    label_data = label_data[0]
    word_label = label_data[:16]
    coords = label_data[16:]
    min_h = coords[0]
    max_h = coords[1]
    min_w = coords[2]
    max_w = coords[3]
    image = np.ones((64, 512, 3), dtype=np.uint8) * 255
    # print(coords)
    image[:, :max_w - min_w, :] = page[min_h: max_h, min_w: max_w, :]

    coder = LabelEncoderDecoder(alphabet='russian')
    print(coder.decode_word(word_label))

    cv2.imshow('img', image)
    cv2.waitKey(0)

    image = image.reshape(1, args.num_of_channels, args.image_height, args.image_width)

    image = Variable(torch.FloatTensor(image))

    model.eval()
    preds = model(image)
    _, preds = preds.max(2)

    preds = preds.transpose(1, 0).contiguous().view(-1)

    print(preds)
    print(coder.decode_word(array=preds))
    print(coder.decode_word(array=coder.from_raw_to_label(array=preds)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs prediction for a given input image.')
    parser.add_argument('--model_path', type=str,
                        default=r'D:\russian_words\models\crnn0.pt',
                        help='Path to a model weights in .pt file')
    parser.add_argument('--data_path', type=str,
                        default=r'D:\russian_words\real\outputs\nabor-texta-3-2.json',
                        help='Path to an .npy file with images')

    parser.add_argument('--image_height', type=int, default=64,
                        help='Height of a single image to perform prediction on.')

    parser.add_argument('--image_width', type=int, default=512,
                        help='Width of a single image to perform prediction on.')

    parser.add_argument('--num_of_channels', type=int, default=3,
                        help='Number of symbols in alphabet including blank character.')

    parser.add_argument('--num_of_classes', type=int, default=33,
                        help='Number of channels in images.')

    parser.add_argument('--num_of_lstm_hidden_units', type=int, default=256,
                        help='Number of LSTM hidden units.')

    _args = parser.parse_args()
    run_test_real(args=_args)
    # run_test_synthetic(args=_args)
