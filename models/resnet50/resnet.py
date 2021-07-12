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
init_bias = 3.


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, name='',  lamba=0.0, temperature=2./3., sparse_in=1):
        super().__init__()

        self.conv1 = MAPConv2d(in_channels, out_channels, kernel_size=1, bias=False,
                               name=name + '.1')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = MAPConv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False,
                               name=name + '.2')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = MAPConv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False,
                               name=name + '.3')
        self.bn3 = nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        self.sc_conv = nn.Sequential()
        self.sc_bn = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.sc_conv = MAPConv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1,
                                     bias=False,
                                     name=name + '.shortcut')
            self.sc_bn = nn.BatchNorm2d(out_channels * BottleNeck.expansion)

        self.sparse = True

        if self.sparse:
            self.gen0 = nn.Linear(sparse_in, out_channels)
            self.sparse1 = SparseChannel(sparse_in, out_channels, lamba=lamba,
                                         temperature=temperature, name=name + '.1', conv=self.conv1)
            self.gen1 = nn.Linear(out_channels, out_channels)
            self.sparse2 = SparseChannel(out_channels, out_channels, lamba=lamba,
                                         temperature=temperature, name=name + '.2', conv=self.conv2)
            nn.init.normal_(self.gen0.bias, init_bias, .01)
            nn.init.normal_(self.gen1.bias, init_bias, .01)
        
    def forward(self, x, logas=None):
        # if self.sparse:
        conv = self.conv1(x)
        conv = self.relu1(self.sparse1(self.bn1(conv), logas[0]))
        conv = self.conv2(conv)
        conv = self.relu2(self.sparse2(self.bn2(conv), logas[1]))
        conv = self.conv3(conv)
        conv = self.bn3(conv)
        sc = self.sc_bn(self.sc_conv(x))
        return nn.ReLU(inplace=True)(conv + sc)


class BlocksContainer(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_blocks, stride, sparse_in, name=None,
                    lamba=0.01, temperature=2. / 3.):
        super().__init__()

        strides = [stride] + [1] * (num_blocks - 1)
        self.fine_tune = False
        self.blocks = nn.Sequential()

        self.generators = nn.Sequential()
        s_in = sparse_in
        for i, stride in enumerate(strides):
            self.blocks.add_module('block%d' % i,
                      block(in_channels, out_channels, stride, name='{}.{}'.format(name, i),
                       lamba=lamba, temperature=temperature, sparse_in=s_in))
            s_in = out_channels
            in_channels = out_channels * block.expansion

    def forward(self, x, logas):
        i = 0
        for layer in self.blocks:
            x = layer(x, logas[i:i+2])
            i += 2
        return x


#ResNet 50 for ImageNet 2012
class ResNet50(nn.Module):

    def __init__(self, block, num_block, num_classes=100, lamba=0.0,
                 temperature=2./3., sparse=True, back_dep=False):
        super().__init__()
        self.threshold = None
        self.in_channels = 64
        self.weight_decay = 0
        self.back_dep = back_dep
        self.sparse = sparse

        self.conv1 = MAPConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, name='conv1')
        self.bn1 = nn.BatchNorm2d(64)
        self.sparse = SparseChannel(3, 64, lamba=lamba,
                                    temperature=temperature, name='conv1', conv=self.conv1)
        self.relu1 = nn.ReLU(inplace=True)
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = BlocksContainer(block, 64, 64, num_block[0], 1, name='conv2', sparse_in=64,
                                       lamba=lamba, temperature=temperature)
        self.conv3_x = BlocksContainer(block, 256, 128, num_block[1], 2, name='conv3', sparse_in=64,
                                        lamba=lamba, temperature=temperature)
        self.conv4_x = BlocksContainer(block, 512, 256, num_block[2], 2, name='conv4', sparse_in=128,
                                       lamba=lamba, temperature=temperature)
        self.conv5_x = BlocksContainer(block, 1024, 512, num_block[3], 2, name='conv5', sparse_in=256,
                                       lamba=lamba, temperature=temperature)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = MAPDense(512 * block.expansion, num_classes, name='fc')

        in_out_forward = [*[(64, 64)] * 7, (64, 128), *[(128, 128)] * 7, (128, 256), *[(256, 256)] * 11, (256, 512),
                          *[(512, 512)] * 5]
        in_out_backward = [*[(512, 512)] * 6, (512, 256), *[(256, 256)] * 11, (256, 128), *[(128, 128)] * 7, (128, 64),
                           *[(64, 64)] * 6]
        gens = []
        for in_c, out_c in (in_out_backward if self.back_dep else in_out_forward):
            gen = nn.Linear(in_c, out_c).cuda()
            nn.init.normal_(gen.bias, 3, .01)
            gens.append(gen)
        self.gens = nn.Sequential(*gens)

        self.conv_layers, self.fc_layers, self.sparse_layers, self.bn_layers, self.bn_params, self.gen_layers = [], [], [], [], [], []
        for m in self.modules():
            if isinstance(m, MAPConv2d):
                self.conv_layers.append(m)
            elif isinstance(m, MAPDense):
                self.fc_layers.append(m)
            elif isinstance(m, SparseChannel):
                self.sparse_layers.append(m)
            elif isinstance(m, nn.Linear):
                self.gen_layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]
                self.bn_layers.append(m)


    def forward(self, x):
        loga = torch.Tensor(self.gens[0].in_features).cuda()
        loga.data.fill_(1)
        logas = []
        for i in range(len(self.gens)):
            loga = self.gens[i](loga).view(-1)
            loga = math.log(100) * torch.tanh(loga)
            logas.append(loga)
        if self.back_dep:
            logas = logas[::-1]

        # output = self.conv1(x)
        conv = self.conv1(x)
        output = self.relu1(self.bn1(conv))
        # sparse the first conv
        output = self.sparse(output, logas[0])
        output = self.maxpool(output)
        output = self.conv2_x(output, logas[1:7])
        output = self.conv3_x(output, logas[7:15])
        output = self.conv4_x(output, logas[15:27])
        output = self.conv5_x(output, logas[27:])
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def fine_tune(self):
        for layer in self.sparse_layers:
            layer.fine_tune = True
            layer.lamba = 0

    def regularization(self):
        regularization = 0
        for layer in self.sparse_layers:
            reg = layer.regularization()
            regularization -= reg
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization


def resnet50(args):
    """ return a ResNet 50 object
    """
    return ResNet50(BottleNeck, [3, 4, 6, 3], num_classes=args.num_classes,
                    lamba=args.lamba, temperature=args.temp, back_dep=args.back_dep)
