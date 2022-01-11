import torch
import torch.nn as nn
from torch.nn import functional as F
import copy


class BatchNorm2d_p(nn.BatchNorm2d):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super(BatchNorm2d_p, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.output = None
        self.input = None
        self.before_affine = None

    def forward(self, input):
        self._check_input_dim(input)

        mean = input.mean([0, 2, 3])[None, :, None, None]
        var = input.var([0, 2, 3], unbiased=False)[None, :, None, None]
        self.input = input        
        self.before_affine = (self.input.clone() - mean)/ torch.sqrt(var + self.eps)        
        self.output = self.before_affine.clone() * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return self.output.clone()

class BatchTransNorm_p(nn.BatchNorm2d):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super(BatchTransNorm_p, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.output = None

    def forward(self, target_input):
        exponential_average_factor = self.momentum

        target_mean = target_input.mean([0, 2, 3])
        target_var = target_input.var([0, 2, 3], unbiased=False)

        n = target_input.numel() / target_input.size(1)
        with torch.no_grad():
            running_target_mean = exponential_average_factor * target_mean\
                + (1 - exponential_average_factor) * self.running_mean
            # update running_var with unbiased var
            running_target_var = exponential_average_factor * target_var * n / (n - 1)\
                + (1 - exponential_average_factor) * self.running_var

            running_target_var = running_target_var[None, :, None, None]
            running_target_mean = running_target_mean[None, :, None, None]

            source_var = self.running_var[None, :, None, None]
            source_mean = self.running_mean[None, :, None, None]

            weight = self.weight[None, :, None, None]
            bias = self.bias[None, :, None, None]

            #  transfer
            target_input = (target_input - running_target_mean) * (torch.sqrt(source_var + self.eps) /
                                                                   torch.sqrt(running_target_var + self.eps)) + source_mean

            transferred_mean = target_input.mean(
                [0, 2, 3])[None, :, None, None]
            transferred_var = target_input.var([0, 2, 3], unbiased=False)[
                None, :, None, None]

            running_transferred_mean = exponential_average_factor * transferred_mean\
                + (1 - exponential_average_factor) * \
                self.running_mean[None, :, None, None]
            # update running_var with unbiased var
            running_transferred_var = exponential_average_factor * transferred_var * n / (n - 1)\
                + (1 - exponential_average_factor) * \
                self.running_var[None, :, None, None]

            target_input = (weight * (target_input - running_transferred_mean) /
                            (torch.sqrt(running_transferred_var + self.eps))) + bias

            self.output = target_input.clone()
            return target_input