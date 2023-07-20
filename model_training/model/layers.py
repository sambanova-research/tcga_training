
"""
Copyright 2023 SambaNova Systems, Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import torch
import torch.nn as nn


class Bias2DConv(nn.Module):
    def __init__(self, num_features):
        super(Bias2DConv, self).__init__()
        self.dummy_conv = nn.Conv2d(num_features, num_features, 1, 1)
        self.dummy_conv.weight = nn.Parameter(torch.eye(num_features).view(num_features, num_features, 1, 1),
                                              requires_grad=False)
        self.dummy_conv.bias.data = torch.zeros_like(self.dummy_conv.bias.data)

    def forward(self, x):
        x = self.dummy_conv(x)
        return x

    def init_pass(self, x, count):
        mu = x.mean(dim=(0, 2, 3)).detach()
        init_mean = self.dummy_conv.bias.data
        init_mean += (mu - init_mean) / (count + 1)
        self.dummy_conv.bias.data = 0 - init_mean
        x = self.dummy_conv(x)
        return x


class Bias2DMean(nn.Module):
    def __init__(self, num_features):
        super(Bias2DMean, self).__init__()
        self.num_features = num_features
        self._bias = nn.Parameter(torch.zeros(num_features, 1, 1))

    def forward(self, x):
        return x + self._bias

    def init_pass(self, x, count):
        mu = x.mean(dim=(0, 2, 3)).detach().view(self._bias.shape)
        init_mean = self._bias.data
        init_mean += (mu - init_mean) / (count + 1)
        self._bias.data += 0 - init_mean
        return x + self._bias


class AvgPool2d(nn.Module):
    def __init__(self, num_features, stride):
        super(AvgPool2d, self).__init__()
        self.dummy_conv = nn.Conv2d(num_features, num_features, stride, stride, bias=False)
        self.dummy_conv.weight = nn.Parameter(torch.eye(num_features).view(num_features, num_features, 1, 1).expand(
            num_features, num_features, stride, stride) / (stride * stride),
                                              requires_grad=False)

    def forward(self, x):
        x = self.dummy_conv(x)
        return x
