"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
import math
from models.l0_layers import MAPConv2d, MAPDense, SparseChannel
cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features
        self.classifier = MAPDense(512, num_class)
        self.layers = [*self.features.layers]
        for layer in self.classifier.modules():
            self.layers.append(layer)
        self.layer_seq = nn.Sequential(*self.layers)


    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output


    def regularization(self):
        regularization = 0.
        in_channel = 3
        for layer in self.features.layers:
            if isinstance(layer, SparseChannel):
                reg, in_channel = layer.regularization(in_channel)
                regularization -= reg
        return regularization

    def fine_tune(self):
        for l in self.layers:
            if isinstance(l, SparseChannel):
                l.fine_tune = True


class LayerContainer(nn.Module):
    def __init__(self, cfg, batch_norm=False, lamba=.1, temperature=2/3., gpu=False):
        super().__init__()
        layers = nn.Sequential()

        input_channel = 3
        self.qz_loga = torch.ones(64)
        if gpu is not None:
            self.qz_loga = self.qz_loga.cuda()
        for i, l in enumerate(cfg):
            if l == 'M':
                layers.add_module('max_pool %d' % i ,nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            conv = MAPConv2d(input_channel, l, kernel_size=3, padding=1, lamba=lamba)
            layers.add_module('conv %d' % i, conv)
            sparse_layer = SparseChannel(input_channel, l, lamba=lamba, temperature=temperature, name=str(i), conv=conv, hc=True, gpu=gpu)

            if batch_norm:
                layers.add_module('batch_norm%d' % i, nn.BatchNorm2d(l))

            layers.add_module('sparse_channel%d' % i, sparse_layer)

            layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
            input_channel = l

        self.layers = layers
        for i in range(len(layers)):
            if gpu is not None:
                layers[i] = layers[i].cuda()
            m = layers[i]
            if isinstance(m, MAPConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MAPDense):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def vgg16_bn(args):
    return VGG(LayerContainer(cfg['D'], batch_norm=True, lamba=args.lamba, temperature=args.temp, gpu=args.gpu),
               num_class=args.num_classes)

