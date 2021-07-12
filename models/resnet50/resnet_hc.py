"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from models.resnet50.l0_layers import MAPConv2d, MAPDense, SparseChannel
import math
limit = math.log(100)


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, name='', droprate_init=0.0, weight_decay=0., sparse=False,
                 lamba=0.01, temperature=2. / 3.):
        super().__init__()

        self.conv1 = MAPConv2d(in_channels, out_channels, kernel_size=1, bias=False,
                               droprate_init=droprate_init, weight_decay=weight_decay)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = MAPConv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False,
                               droprate_init=droprate_init, weight_decay=weight_decay, name=name + '.2')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = MAPConv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False,
                               droprate_init=droprate_init, weight_decay=weight_decay, name=name + '.3')
        self.bn3 = nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        self.sc_conv = nn.Sequential()
        self.sc_bn = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.sc_conv = MAPConv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1,
                                     bias=False, droprate_init=droprate_init, weight_decay=weight_decay,
                                     name=name + '.shortcut')
            self.sc_bn = nn.BatchNorm2d(out_channels * BottleNeck.expansion)

        self.gen0 = nn.Linear(out_channels, out_channels)
        self.sparse1 = SparseChannel(in_channels, out_channels, droprate_init=droprate_init, lamba=lamba,
                                     temperature=temperature, name=name + '.1', conv=self.conv1, hc=True)
        self.sparse2 = SparseChannel(out_channels, out_channels, droprate_init=droprate_init, lamba=lamba,
                                     temperature=temperature, name=name + '.2', conv=self.conv2, hc=True)
        # self.sparse3 = SparseChannel(out_channels, out_channels * BottleNeck.expansion, droprate_init=droprate_init,
        #                              lamba=lamba,
        #                              temperature=temperature, name=name + '.3', conv=self.conv3)
        
    def forward(self, x):
        conv = self.conv1(x)
        conv = self.sparse1(self.relu1(self.bn1(conv)))
        conv = self.conv2(conv)
        conv = self.sparse2(self.relu2(self.bn2(conv)))
        conv = self.conv3(conv)
        conv = self.bn3(conv)
        sc = self.sc_bn(self.sc_conv(x))
        return nn.ReLU(inplace=True)(conv + sc)


class BlocksContainer(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_blocks, stride, name=None,
                 lamba=0.0, temperature=2. / 3., sparse=False):
        super().__init__()

        strides = [stride] + [1] * (num_blocks - 1)
        self.layers = nn.Sequential()
        for i, stride in enumerate(strides):
            self.layers.add_module('block%d' % i,
                block(in_channels, out_channels, stride, name='{}.{}'.format(name, i),
                      lamba=lamba, temperature=temperature, sparse=sparse))
            in_channels = out_channels * block.expansion


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


#ResNet 50 for ImageNet 2012
class ResNet50(nn.Module):

    def __init__(self, block, num_block, lamba=0.01, temperature=2./3.):
        super().__init__()
        self.threshold = None
        self.in_channels = 64
        self.weight_decay = 0

        self.conv1 = MAPConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, name='conv1')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.sparse1 = SparseChannel(3, 64, lamba=lamba,
                                     temperature=temperature, name='conv1', conv=self.conv1, hc=True)
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = BlocksContainer(block, 64, 64, num_block[0], 1, name='conv2',
                                        lamba=lamba, temperature=temperature)
        self.conv3_x = BlocksContainer(block, 256, 128, num_block[1], 2, name='conv3',
                                        lamba=lamba, temperature=temperature)
        self.conv4_x = BlocksContainer(block, 512, 256, num_block[2], 2, name='conv4',
                                       lamba=lamba, temperature=temperature)
        self.conv5_x = BlocksContainer(block, 1024, 512, num_block[3], 2, name='conv5',
                                       lamba=lamba, temperature=temperature)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MAPDense(512 * block.expansion, 1000, name='fc')
        self.conv_layers, self.fc_layers, self.sparse_layers, self.bn_layers, self.bn_params = [], [], [], [], []
        for m in self.modules():
            if isinstance(m, MAPConv2d):
                self.conv_layers.append(m)
            elif isinstance(m, MAPDense):
                self.fc_layers.append(m)
            elif isinstance(m, SparseChannel):
                self.sparse_layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]
                self.bn_layers.append(m)

    def fine_tune(self):
        for layer in self.sparse_layers:
            layer.fine_tune = True
            layer.lamba = 0

    def forward(self, x):
        # output = self.conv1(x)
        conv = self.conv1(x)
        output = self.relu1(self.bn1(conv))
        output = self.maxpool(output)
        output = self.sparse1(output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def regularization(self):
        regularization = 0.
        for layer in self.sparse_layers:
            regularization -= layer.regularization()
        for bnw in self.bn_params:
            if self.weight_decay > 0:
                regularization += self.weight_decay * .5 * torch.sum(bnw.pow(2))
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization


# ResNet56 for CIFAR dataset

def resnet50(args):
    """ return a ResNet 50 object
    """
    return ResNet50(BottleNeck, [3, 4, 6, 3], lamba=args.lamba, temperature=args.temp)



