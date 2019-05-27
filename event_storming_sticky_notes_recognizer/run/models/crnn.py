import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=nIn, hidden_size=nHidden, bidirectional=True)
        self.embedding = nn.Linear(in_features=nHidden * 2, out_features=nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):

    def __init__(self, image_height, num_of_channels, num_of_classes, num_of_lstm_hidden_units, n_rnn=2,
                 leakyRelu=False):
        super(CRNN, self).__init__()
        assert image_height % 16 == 0, 'imgH has to be a multiple of 16'

        kernel_sizes = [3, 3, 3, 3, 3, 3, (4, 1)]
        pads = [1, 1, 1, 1, 1, 1, 0]
        strides = [1, 1, 1, 1, 1, 1, 1]
        nums_of_filters = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = num_of_channels if i == 0 else nums_of_filters[i - 1]
            nOut = nums_of_filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(in_channels=nIn,
                                     out_channels=nOut,
                                     kernel_size=kernel_sizes[i],
                                     stride=strides[i],
                                     padding=pads[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0, batchNormalization=True)  # 64x64x512
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x32x256
        convRelu(1, batchNormalization=True)  # 128x32x256
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x16x128
        convRelu(2, batchNormalization=True)  # 256x16x128
        convRelu(3, batchNormalization=True)  # 256x16x128
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((1, 2), (2, 2)))  # 256x8x64
        convRelu(4, batchNormalization=True)  # 512x8x64
        convRelu(5, batchNormalization=True)  # 512x8x64
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((1, 2), (2, 2)))  # 512x4x32
        convRelu(6, batchNormalization=True)  # 512x1x32

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, num_of_lstm_hidden_units, num_of_lstm_hidden_units),
            BidirectionalLSTM(num_of_lstm_hidden_units, num_of_lstm_hidden_units, num_of_classes))

        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        output = self.softmax(output)

        return output
