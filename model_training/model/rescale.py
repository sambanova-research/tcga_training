
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
import torch.nn.functional as F

from .layers import Bias2DMean


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 block_idx: int,
                 max_block: int,
                 stride: int = 1,
                 groups: int = 1,
                 base_width: int = 64,
                 drop_conv=0.0) -> None:

        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, groups=groups, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, groups=groups, bias=False)

        self.addbias1 = Bias2DMean(inplanes)
        self.addbias2 = Bias2DMean(planes)

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.stride = stride
        self._scale = nn.Parameter(torch.ones(1))
        multiplier = (block_idx + 1)**-(1 / 6) * max_block**(1 / 6)
        multiplier = multiplier * (1 - drop_conv)**.5

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                _, C, H, W = m.weight.shape
                stddev = (C * H * W / 2)**-.5
                nn.init.normal_(m.weight, std=stddev * multiplier)

        self.residual = max_block**-.5
        self.identity = block_idx**.5 / (block_idx + 1)**.5

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            if stride == 1:
                avgpool = nn.Sequential()
            else:
                avgpool = nn.AvgPool2d(stride)

            self.downsample = nn.Sequential(avgpool, Bias2DMean(num_features=inplanes),
                                            nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, bias=False))

            nn.init.kaiming_normal_(self.downsample[2].weight, a=1)

        self.drop = nn.Sequential()
        if drop_conv > 0.0:
            self.drop = nn.Dropout2d(drop_conv)

    def forward(self, x):
        # Not adding dropout here.
        out = F.relu(self.drop(self.conv1(self.addbias1(x))))
        out = self.drop(self.conv2(self.addbias2(out)))
        out = out * self.residual * self._scale + self.identity * self.downsample(x)
        out = F.relu(out)
        return out

    def init_pass(self, x, count):
        out = F.relu(self.drop(self.conv1(self.addbias1.init_pass(x, count))))
        out = self.drop(self.conv2(self.addbias2.init_pass(out, count)))
        out = out * self.residual * self._scale + self.identity * self.downsample(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, block_idx, max_block, stride=1, groups=1, base_width=64, drop_conv=0.0):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, stride=stride, groups=groups, bias=False)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)

        self.addbias1 = Bias2DMean(inplanes)
        self.addbias2 = Bias2DMean(width)
        self.addbias3 = Bias2DMean(width)

        self._scale = nn.Parameter(torch.ones(1))
        multiplier = (block_idx + 1)**-(1 / 6) * max_block**(1 / 6)
        multiplier = multiplier * (1 - drop_conv)**.5

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                _, C, H, W = m.weight.shape
                stddev = (C * H * W / 2)**-.5
                nn.init.normal_(m.weight, std=stddev * multiplier)

        self.residual = max_block**-.5
        self.identity = block_idx**.5 / (block_idx + 1)**.5

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            if stride == 1:
                avgpool = nn.Sequential()
            else:
                avgpool = nn.AvgPool2d(stride)

            self.downsample = nn.Sequential(avgpool, Bias2DMean(num_features=inplanes),
                                            nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, bias=False))
            nn.init.kaiming_normal_(self.downsample[2].weight, a=1)

        self.drop = nn.Sequential()
        if drop_conv > 0.0:
            self.drop = nn.Dropout2d(drop_conv)

    def forward(self, x):
        out = F.relu(self.drop(self.conv1(self.addbias1(x))))
        out = F.relu(self.drop(self.conv2(self.addbias2(out))))
        out = self.drop(self.conv3(self.addbias3(out)))
        out = out * self.residual * self._scale + self.identity * self.downsample(x)
        out = F.relu(out)
        return out

    def init_pass(self, x, count):
        out = F.relu(self.drop(self.conv1(self.addbias1.init_pass(x, count))))
        out = F.relu(self.drop(self.conv2(self.addbias2.init_pass(out, count))))
        out = self.drop(self.conv3(self.addbias3.init_pass(out, count)))
        out = out * self.residual * self._scale + self.identity * self.downsample(x)
        out = F.relu(out)
        return out


class ReScale(nn.Module):
    def __init__(self,
                 layers,
                 num_classes=1000,
                 groups=1,
                 width_per_group=64,
                 drop_conv=0.0,
                 drop_fc=0.0,
                 block=Bottleneck,
                 input_shapes=(None, None),
                 num_flexible_classes=-1):
        super(ReScale, self).__init__()

        self.inplanes = 64
        self.num_classes = num_classes
        self.input_shapes = input_shapes
        self.groups = groups
        self.base_width = width_per_group
        self.block_idx = sum(layers) - 1
        self.max_depth = sum(layers)
        self.num_flexible_classes = num_flexible_classes

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.addbias1 = Bias2DMean(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_conv=drop_conv)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, drop_conv=drop_conv)
        self.addbias2 = Bias2DMean(512 * block.expansion)
        self.drop = nn.Dropout(drop_fc)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.mean_pool = nn.AvgPool2d((input_shapes[0] // 32, input_shapes[1] // 32))

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.fc.weight, a=1)

        if self.num_flexible_classes != -1:
            _fixed_sum_layer = torch.zeros(num_classes)
            num_unused_classes = num_classes - self.num_flexible_classes
            if num_unused_classes > 0:
                _fixed_sum_layer[self.num_flexible_classes:] = torch.ones(num_unused_classes) * -10000.0
                # initialize bias and weight of unused to 0
                self.fc.bias.data[self.num_flexible_classes:] = 0
                self.fc.weight.data[self.num_flexible_classes:, :] = 0

            # make the fixed_mask not trainable
            self.register_buffer("fixed_sum_layer", _fixed_sum_layer)

    def _make_layer(self, block, planes, num_blocks, stride=1, drop_conv=0.0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.inplanes,
                      planes,
                      block_idx=self.block_idx,
                      max_block=self.max_depth,
                      stride=stride,
                      groups=self.groups,
                      base_width=self.base_width,
                      drop_conv=drop_conv))
            self.inplanes = planes * block.expansion
            self.block_idx += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.addbias1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.addbias2(x)

        x = self.mean_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.drop(x)
        x = self.fc(x)
        if self.num_flexible_classes != -1:
            x = x + self.fixed_sum_layer

        return x

    def layer_init_pass(self, layer, x, count):
        for each in layer:
            x = each.init_pass(x, count)
        return x

    def init_pass(self, x, count):
        x = self.conv1(x)
        x = self.addbias1.init_pass(x, count)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer_init_pass(self.layer1, x, count)
        x = self.layer_init_pass(self.layer2, x, count)
        x = self.layer_init_pass(self.layer3, x, count)
        x = self.layer_init_pass(self.layer4, x, count)
        x = self.addbias2.init_pass(x, count)
        x = torch.mean(torch.mean(x, -1), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x


def rescale18(num_classes=1000, drop_conv=0.0, drop_fc=0.0, **kwargs):
    return ReScale([2, 2, 2, 2],
                   num_classes=num_classes,
                   drop_conv=drop_conv,
                   drop_fc=drop_fc,
                   groups=1,
                   width_per_group=64,
                   block=BasicBlock,
                   **kwargs)


def rescale50(num_classes=1000, drop_conv=0.0, drop_fc=0.0, **kwargs):
    return ReScale([3, 4, 6, 3],
                   num_classes=num_classes,
                   drop_conv=drop_conv,
                   drop_fc=drop_fc,
                   groups=1,
                   width_per_group=64,
                   **kwargs)


def rescale101(num_classes=1000, drop_conv=0.0, drop_fc=0.0, **kwargs):
    return ReScale([3, 4, 23, 3],
                   num_classes=num_classes,
                   drop_conv=drop_conv,
                   drop_fc=drop_fc,
                   groups=1,
                   width_per_group=64,
                   **kwargs)


def rescale200(num_classes=1000, drop_conv=0.0, drop_fc=0.0, **kwargs):
    return ReScale([3, 24, 36, 3],
                   num_classes=num_classes,
                   drop_conv=drop_conv,
                   drop_fc=drop_fc,
                   groups=1,
                   width_per_group=64,
                   **kwargs)


def rescaleX50_32x4d(num_classes=1000, drop_conv=0.0, drop_fc=0.0, **kwargs):
    return ReScale([3, 4, 6, 3],
                   num_classes=num_classes,
                   drop_conv=drop_conv,
                   drop_fc=drop_fc,
                   groups=32,
                   width_per_group=4,
                   **kwargs)


def rescaleX101_32x8d(num_classes=1000, drop_conv=0.0, drop_fc=0.0, **kwargs):
    return ReScale([3, 4, 23, 3],
                   num_classes=num_classes,
                   drop_conv=drop_conv,
                   drop_fc=drop_fc,
                   groups=32,
                   width_per_group=8,
                   **kwargs)
