import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class MAPDense(Module):

    def __init__(self, in_features, out_features, bias=True, weight_decay=0., name='', **kwargs):
        super(MAPDense, self).__init__()
        self.layer_name = name
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weight, mode='fan_out')

        if self.bias is not None:
            self.bias.data.normal_(0, 1e-2)

    def _reg_w(self, **kwargs):
        logpw = - torch.sum(self.weight_decay * .5 * (self.weight.pow(2)))
        logpb = 0
        if self.bias is not None:
            logpb = - torch.sum(self.weight_decay * .5 * (self.bias.pow(2)))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def flops_params(self):
        expected_flops = self.in_features * self.out_features
        expected_params = self.in_features * self.out_features
        if self.bias is not None:
            expected_params += self.out_features
        return torch.tensor([expected_flops, expected_params])

    def forward(self, input):
        output = input.mm(self.weight)
        if self.bias is not None:
            output.add_(self.bias.view(1, self.out_features).expand_as(output))
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', weight_decay: ' \
            + str(self.weight_decay) + ')'


class MAPConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 weight_decay=0., name='', **kwargs):
        super(MAPConv2d, self).__init__()
        self.weight_decay = weight_decay
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.input_shape = None
        self.layer_name = name
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weight, mode='fan_in')
        if self.bias is not None:
            self.bias.data.normal_(0, 1e-2)

    def _reg_w(self, **kwargs):
        return 0

    def regularization(self):
        return self._reg_w()

    def forward(self, input_):
        if self.input_shape is None:
            self.input_shape = input_.size()
        output = F.conv2d(input_, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def __repr__(self):
        s = ('{name}{layer_name}({in_channels}, {out_channels}, kernel_size={kernel_size} '
             ', stride={stride}, weight_decay={weight_decay}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class SparseChannel(Module):
    """Implementation of L0 Sparse Channel"""
    def __init__(self, in_channels, out_channels, lamba=5e-4, temperature=2./3., local_rep=False, name='',
                 conv=None, hc=False, gpu=False, **kwargs):
        """
        :param out_channels: Number of output channels
        """
        super(SparseChannel, self).__init__()
        self.layer_name = name
        self.in_channels = in_channels
        self.ppos = self.out_channels = out_channels
        self.temperature = temperature
        self.floatTensor = torch.FloatTensor if gpu is None else torch.cuda.FloatTensor
        self.use_bias = False
        self.dim_z = out_channels
        self.input_shape = None
        self.local_rep = local_rep
        self.prior_prec = 0
        self.lamba = lamba
        self.conv = conv
        self.fine_tune = False
        if hc:
            self.qz_loga = Parameter(torch.Tensor(out_channels))
            self.qz_loga.data.normal_(3, 1e-2)

        print(self)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self, in_channel):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        w = self.conv.kernel_size[0] * self.conv.kernel_size[1]
        logpw_col = torch.sum(- (.5 * self.conv.weight_decay * self.conv.weight.pow(2)) - self.lamba, 3).sum(2).sum(1)
        return - in_channel * torch.sum((1 - q0)) * w * self.lamba, torch.sum((1 - q0))

    def regularization(self, in_channel):
        return self._reg_w(in_channel)

    def flops_params(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        if self.training:
            self.ppos = ppos = torch.sum(1 - self.cdf_qz(0))
        else:
            z = self.sample_z(1, sample=self.training, )
            self.ppos = ppos = (z > 0).sum()
        n = self.conv.kernel_size[0] * self.conv.kernel_size[1] * self.conv.in_channels  # vector_length
        flops_per_instance = n  # (n: multiplications and n-1: additions)

        num_instances_per_filter = ((self.conv.input_shape[2] - self.conv.kernel_size[0] + 2 * self.conv.padding[0]) / self.conv.stride[0]) + 1  # for rows
        num_instances_per_filter *= ((self.conv.input_shape[3] - self.conv.kernel_size[1] + 2 * self.conv.padding[1]) / self.conv.stride[1]) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_params = n * ppos

        if self.use_bias:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_params += ppos

        return torch.Tensor([expected_flops, expected_params])

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if self.fine_tune:
            pi = torch.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1).detach()

        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.dim_z))
            z = self.quantile_concrete(eps).view(batch_size, self.dim_z, 1, 1)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def forward(self, input_, qz_loga=None,):
        if qz_loga is not None:
            self.qz_loga = qz_loga
        else:
            self.loga = math.log(100) * torch.tanh(self.qz_loga)
        if self.input_shape is None:
            self.input_shape = input_.size()
        # if self.local_rep or not self.training:
        z = self.sample_z(input_.size(0), sample=self.training,)
        self.z = z
        return input_.mul(z)

    def __repr__(self):
        return 'SparseChannel lambda={}'.format(self.lamba)



