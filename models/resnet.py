"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from models.l0_layers import MAPConv2d, MAPDense, SparseChannel
import math
limit = math.log(100)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, name='',
                lamba=0.01, temperature=2./3., gpu=False):
        super().__init__()

        self.conv1 = MAPConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                      name=name+'.1')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = MAPConv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False,
                      name=name+'.2')
        self.bn2 = nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        #shortcut
        self.shortcut = nn.Sequential()
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                MAPConv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False,
                          name=name+'shortcut'),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.sparse = SparseChannel(in_channels, out_channels, lamba=lamba,
                                         temperature=temperature, name=name + '.1', conv=self.conv1, gpu=gpu)

    def forward(self, x, loga=None):
        conv = self.conv1(x)
        conv = self.sparse(self.relu1(self.bn1(conv)), loga)
        conv = self.bn2(self.conv2(conv))
        sc = self.shortcut(x)
        return nn.ReLU(inplace=True)(conv + sc)


class BlocksContainer(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_blocks, stride, name=None,
                   lamba=0.01, temperature=2. / 3., gpu=False):
        super().__init__()

        strides = [stride] + [1] * (num_blocks - 1)
        self.fine_tune = False
        self.blocks = nn.Sequential()
        self.generators = nn.Sequential()

        for i, stride in enumerate(strides):
            self.blocks.add_module('block%d' % i,
                    block(in_channels, out_channels, stride, name='{}.{}'.format(name, i),
                     lamba=lamba, temperature=temperature, gpu=gpu))
            gen = nn.Linear(in_channels, out_channels)
            if gpu is not None:
                gen = gen.cuda()
            in_channels = out_channels * block.expansion
            nn.init.normal_(gen.bias, 3, .01)
            self.generators.add_module('gen%d' % i, gen)

    def flops_params(self):
        f_p = torch.zeros(2)
        for l in self.blocks:
            f_p += l.flops_params()
        return f_p


    def forward(self, x, logas):
        for layer, loga in zip(self.blocks, logas):
            x = layer(x, loga)
        return x



# ResNet56 for CIFAR dataset
class ResNet56(nn.Module):

    def __init__(self, block, num_block, num_classes=100,
                  lamba=0.01, temperature=2./3., back_dep=False, gpu=False):
        super().__init__()
        self.threshold = None
        self.in_channels = 16
        self.gpu = gpu
        self.conv1 = MAPConv2d(3, 16, kernel_size=3, padding=1, stride=1, bias=False, name='conv1')
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.sparse = SparseChannel(3, 16,  lamba=lamba,
                                     temperature=temperature, name='conv1', conv=self.conv1, gpu=gpu)
        self.back_dep = back_dep

        in_out_forward = [*[(16, 16)]*10, (16,32), *[(32,32)]*8, (32,64), *[(64,64)]*8]
        in_out_backward = [*[(64,64)]*9, (64,32), *[(32,32)]*8, (32,16), *[(16,16)]*9]
        gens = []

        for in_c, out_c in (in_out_backward if self.back_dep else in_out_forward):
            gen = nn.Linear(in_c, out_c)
            if gpu is not None:
                gen = gen.cuda()
            nn.init.normal_(gen.bias, 3, .01)
            gens.append(gen)

        self.gens = nn.Sequential(*gens)
        self.conv2_x = BlocksContainer(block, 16, 16, num_block[0], 1, name='conv2',
                                       lamba=lamba, temperature=temperature, gpu=gpu)
        self.conv3_x = BlocksContainer(block, 16, 32, num_block[1], 2, name='conv3',
                                       lamba=lamba, temperature=temperature, gpu=gpu)
        self.conv4_x = BlocksContainer(block, 32, 64, num_block[2], 2, name='conv4',
                                       lamba=lamba, temperature=temperature, gpu=gpu)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MAPDense(64, num_classes, name='fc')
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
        for l in self.sparse_layers:
            l.fine_tune = True

    def forward(self, x):

        loga = torch.Tensor(self.gens[0].in_features)
        if self.gpu is not None:
            loga = loga.cuda()
        loga.data.fill_(1)
        logas = []
        for i in range(len(self.gens)):
            loga = self.gens[i](loga).view(-1)
            loga = math.log(100) * torch.tanh(loga)
            logas.append(loga)
        if self.back_dep:
            logas = logas[::-1]
        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.sparse(output, logas[0])
        output = self.conv2_x(output, logas[1:10])
        output = self.conv3_x(output, logas[10:19])
        output = self.conv4_x(output, logas[19:])
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def regularization(self):
        regularization = 0.
        in_channel = 3
        for layer in self.sparse_layers:
            reg, in_channel = layer.regularization(in_channel)
            regularization -= reg
        if self.gpu is not None:
            regularization = regularization.cuda()
        return regularization


def resnet56(args):
    return ResNet56(BasicBlock, [9, 9, 9], num_classes=args.num_classes,
                 lamba=args.lamba, temperature=args.temp, back_dep=args.back_dep, gpu=args.gpu)
