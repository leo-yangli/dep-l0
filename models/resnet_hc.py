import torch
import torch.nn as nn
from models.l0_layers import MAPConv2d, MAPDense, SparseChannel
import math
limit = math.log(100)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, name='', droprate_init=0.0, weight_decay=0.,
                  lamba=0.01, temperature=2./3., gpu=False):
        super().__init__()

        self.conv1 = MAPConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                     droprate_init=droprate_init, weight_decay=weight_decay, name=name+'.1')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = MAPConv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False,
                     droprate_init=droprate_init, weight_decay=weight_decay, name=name+'.2')
        self.bn2 = nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        #shortcut
        self.shortcut = False

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = True
            self.sc_conv = MAPConv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False,
                         droprate_init=droprate_init, weight_decay=weight_decay, name=name+'shortcut')
            self.sc_bn = nn.BatchNorm2d( out_channels * BasicBlock.expansion)

        self.sparse1 = SparseChannel(in_channels, out_channels, droprate_init=droprate_init, lamba=lamba,
                                         temperature=temperature, name=name + '.1', conv=self.conv1, hc=True, gpu=gpu)

    def forward(self, x):
        conv = self.conv1(x)
        conv = self.relu1(self.sparse1(self.bn1(conv)))
        conv = self.conv2(conv)
        conv = self.bn2(conv)
        if self.shortcut:
            conv += self.sc_bn(self.sc_conv(x))
        else:
            conv += x
        return nn.ReLU(inplace=True)(conv)



class BlocksContainer(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_blocks, stride, name=None, droprate_init=0.3,
                    weight_decay=5e-4, lamba=0.0, temperature=2. / 3., gpu=False):
        super().__init__()

        strides = [stride] + [1] * (num_blocks - 1)
        self.layers = nn.Sequential()
        qz_loga = nn.Parameter(torch.Tensor(out_channels))
        if gpu is not None:
            self.qz_loga = qz_loga.cuda()
        else:
            self.qz_loga = qz_loga
        self.fine_tune = False
        self.qz_loga.data.normal_(math.log(1 - droprate_init) - math.log(droprate_init), 1e-2)
        self.qz_loga.data.normal_(0, 2)
        self.qz_loga.data.fill_(1)
        for i, stride in enumerate(strides):
            self.layers.add_module('block%d' % i,
                block(in_channels, out_channels, stride, name='{}.{}'.format(name, i), droprate_init=droprate_init,
                      weight_decay=weight_decay, lamba=lamba, temperature=temperature,  gpu=gpu))
            in_channels = out_channels * block.expansion


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ResNet56 for CIFAR dataset
class ResNet56(nn.Module):

    def __init__(self, block, num_block, num_classes=100, droprate_init=0.3,
                 weight_decay=5e-4, local_rep=False, lamba=0.01, temperature=2./3.,  gpu=False):
        super().__init__()
        self.threshold = None
        self.in_channels = 16
        self.weight_decay = 0

        self.conv1 = MAPConv2d(3, 16, kernel_size=3, padding=1, stride=1, bias=False, name='conv1')
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.sparse = SparseChannel(3, 16, droprate_init=droprate_init, lamba=lamba,
                                     temperature=temperature, name='conv1', conv=self.conv1, hc=True, gpu=gpu)
        self.conv2_x = BlocksContainer(block, 16, 16, num_block[0], 1, name='conv2', droprate_init=droprate_init,
                                       weight_decay=weight_decay,  lamba=lamba, temperature=temperature,  gpu=gpu)
        self.conv3_x = BlocksContainer(block, 16, 32, num_block[1], 2, name='conv3', droprate_init=droprate_init,
                                       weight_decay=weight_decay, lamba=lamba, temperature=temperature, gpu=gpu)
        self.conv4_x = BlocksContainer(block, 32, 64, num_block[2], 2, name='conv4', droprate_init=droprate_init,
                                       weight_decay=weight_decay, lamba=lamba, temperature=temperature,  gpu=gpu)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MAPDense(64, num_classes, name='fc')
        self.conv_layers, self.fc_layers, self.sparse_layers, self.bn_layers, self.bn_params = [], [], [], [], []
        self.gpu = gpu
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
        for l in self.layers:
            l.fine_tune = True

    def forward(self, x):
        conv = self.conv1(x)
        output = self.relu1(self.bn1(conv))
        output = self.sparse(output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def regularization(self):
        regularization = 0.
        in_channel = 3
        for layer in self.sparse_layers:
            if isinstance(layer, SparseChannel):
                reg, in_channel = layer.regularization(in_channel)
                regularization -= reg
        for bnw in self.bn_params:
            if self.weight_decay > 0:
                regularization -= self.weight_decay* .5 * torch.sum(bnw.pow(2))
        if self.gpu is not None:
            regularization = regularization.cuda()
        return regularization


def resnet56(args):
    return ResNet56(BasicBlock, [9, 9, 9], num_classes=args.num_classes,
                  weight_decay=args.weight_decay, local_rep=False, lamba=args.lamba, temperature=args.temp, gpu=args.gpu)

