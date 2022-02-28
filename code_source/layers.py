__author__ = 'marvinler'

import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.modules.utils import _single, _pair, _triple


# from torch._jit_internal import weak_module, weak_script_method


# @weak_module
class PolarConvNd(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, dimensions=2, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        self.init_kernel_size = kernel_size
        assert kernel_size % 2 == 1, 'expected kernel size to be odd, found %d' % kernel_size
        self.init_dimensions = dimensions

        self.base_vectors = torch.from_numpy(self.build_base_vectors()).float()
        self.true_base_vectors_shape = self.base_vectors.shape
        self.base_vectors = self.base_vectors.view(self.true_base_vectors_shape[0],
                                                   np.prod(self.true_base_vectors_shape[1:]).astype(int))

        inferred_kernel_size = self.true_base_vectors_shape[0]
        _kernel_size = _single(inferred_kernel_size)
        _stride = _single(stride)
        _padding = _single(padding)
        _dilation = _single(dilation)
        super(PolarConvNd, self).__init__(
            in_channels, out_channels, _kernel_size, _stride, _padding, _dilation,
            False, _single(0), groups, bias, padding_mode)

        if dimensions == 2:
            self.reconstructed_stride = _pair(stride)
            self.reconstructed_padding = _pair(padding)
            self.reconstructed_dilation = _pair(dilation)
            self.reconstructed_conv_op = F.conv2d
        elif dimensions == 3:
            self.reconstructed_stride = _triple(stride)
            self.reconstructed_padding = _triple(padding)
            self.reconstructed_dilation = _triple(dilation)
            self.reconstructed_conv_op = F.conv3d
        else:
            raise ValueError('dimension %d not supported' % dimensions)

    def build_base_vectors(self):
        kernel_size = self.init_kernel_size
        middle = kernel_size // 2
        dimensions = self.init_dimensions

        base_vectors = []
        # Burning phase: determine the number of base vectors
        unique_distances = []
        if dimensions == 2:
            for i in range(kernel_size):
                for j in range(kernel_size):
                    i_ = abs(i - middle)
                    j_ = abs(j - middle)
                    unique_distances.append(int(i_ * i_ + j_ * j_))
        elif dimensions == 3:
            for i in range(kernel_size):
                for j in range(kernel_size):
                    for k in range(kernel_size):
                        i_ = abs(i - middle)
                        j_ = abs(j - middle)
                        k_ = abs(k - middle)
                        unique_distances.append(int(i_ * i_ + j_ * j_ + k_ * k_))
        unique_distances, distances_counts = np.unique(unique_distances, return_counts=True)
        unique_distances = np.sort(unique_distances)
        print(*zip(unique_distances, distances_counts), len(unique_distances))

        for unique_distance, n in zip(unique_distances, distances_counts):  # number of base vectors
            base_vector = np.zeros([kernel_size] * dimensions)
            if dimensions == 2:
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        i_ = abs(i - middle)
                        j_ = abs(j - middle)
                        if int(i_ * i_ + j_ * j_) == unique_distance:
                            base_vector[i, j] = 1./n
            elif dimensions == 3:
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        for k in range(kernel_size):
                            i_ = abs(i - middle)
                            j_ = abs(j - middle)
                            k_ = abs(k - middle)
                            if int(i_ * i_ + j_ * j_ + k_ * k_) == unique_distance:
                                base_vector[i, j, k] = 1./n
            base_vectors.append(base_vector)
        base_vectors = np.asarray(base_vectors)
        return base_vectors

    # @weak_script_method
    def forward(self, input):
        # print('old weight', self.weight.size())
        # print('shape self.weight[..., [0]]', self.weight[..., [0]].size())
        # weight = self.weight[..., [0]] * self.a_
        # weight = torch.dot(self.weight[..., [0]], self.a_)
        # print('self.weight[..., 0].view(*self.weight.size()[:-1], 1, 1)', self.weight[..., 0].view(*self.weight.size()[:-1], 1, 1).size())

        # Reconstruct 2D filters
        # print('self.weight.shape', self.weight.shape, 'self.base_vectors.shape', self.base_vectors.shape)
        # print('self.weight[..., 0].view(*self.weight.size()[:-1], 1, 1).shape',
        #       self.weight[..., 0].view(*self.weight.size()[:-1], 1, 1).shape)
        # print('self.weight[..., 0].view(*self.weight.size()[:-1], 1, 1) * self.a_.shape',
        #       (self.weight[..., 0].view(*self.weight.size()[:-1], 1, 1) * self.a_).shape)
        # print('self.weight[..., 0].view(*self.weight.size()[:-1], *[1]*self.init_dimensions).shape',
        #       self.weight[..., 0].view(*self.weight.size()[:-1], *[1]*self.init_dimensions).shape)
        # print('self.weight.view(*self.weight.size()[:-1], *[1]*self.init_dimensions, self.weight.shape[-1]).shape',
        #       self.weight.view(*self.weight.size()[:-1], *[1]*self.init_dimensions, self.weight.shape[-1]).shape)
        # print(self.weight)
        # print(self.weight.view(*self.weight.size()[:-1], *[1]*self.init_dimensions, self.weight.shape[-1]))
        #
        # a = self.weight.shape[0]
        # b = self.weight.shape[1]
        # c = self.weight.shape[2]
        # d = self.base_vectors.shape[0]
        # e = self.base_vectors.shape[1]
        # f = self.base_vectors.shape[2]
        # assert c == d
        # print('self.weight.view(a*b, c).shape', self.weight.view(a*b, c).shape)
        # print('self.base_vectors.view(c, e*f).shape', self.base_vectors.view(c, e*f).shape)
        # print(torch.mm(self.weight.view(a*b, c), self.base_vectors.view(c, e*f)).view(a, b, e, f).shape)
        # print('\n'*5)
        # weight = self.weight[..., 0].view(*self.weight.size()[:-1], 1, 1) * self.a_ + \
        #          self.weight[..., 1].view(*self.weight.size()[:-1], 1, 1) * self.b_ + \
        #          self.weight[..., 2].view(*self.weight.size()[:-1], 1, 1) * self.c_

        #print(input.size())
        #input = torch.cat((input,input,input),1)
        #print(input.size())


        weight_size = self.weight.shape
        weight = torch.mm(self.weight.view(np.prod(weight_size[:-1]), weight_size[-1]), self.base_vectors) \
            .view(*weight_size[:-1], *self.true_base_vectors_shape[1:])
        return self.reconstructed_conv_op(input, weight, self.bias, self.reconstructed_stride,
                                          self.reconstructed_padding, self.reconstructed_dilation, self.groups)


    #torch.repeat_interleave(input, 1, dim=1)
    # def cuda(self, device=None):
    #     self.base_vectors = self.base_vectors.cuda(device)
    #     super(PolarConvNd, self).cuda(device)


    def __repr__(self):
        return ('PolarConv%dd' % self.init_dimensions) + '(' + self.extra_repr() + ')'


if __name__ == '__main__':
    kernel_size = int(sys.argv[-1])
    dimensions = int(sys.argv[-2])
    a = PolarConvNd(in_channels=2, out_channels=4,
                    kernel_size=kernel_size,
                    dimensions=dimensions).to('cuda')
    # print(a.base_vectors)
    to_torch = lambda array: torch.from_numpy(array).to('cuda').float()
    z = np.zeros((1, 2, kernel_size, kernel_size))
    one = np.copy(z)
    two = np.copy(z)
    thr = np.copy(z)
    fou = np.copy(z)
    one[..., -1] = 1
    two[..., 0] = 1
    thr[..., 0, :] = 1
    fou[..., -1, :] = 1

    print(a(to_torch(one)))
    print(a(to_torch(two)))
    print(a(to_torch(thr)))
    print(a(to_torch(fou)))
