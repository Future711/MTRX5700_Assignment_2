#!/usr/bin/env python
# coding: utf-8

"""
ResNet architectures for traffic sign classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard ResNet residual block with two 3x3 convolutions.

    Each block computes:
        F(x) = Conv -> BN -> ReLU -> Conv -> BN
    and adds a shortcut connection:
        out = ReLU(F(x) + shortcut(x))

    When the spatial resolution is halved (stride=2) or the channel count
    changes, a 1x1 projection convolution is used in the shortcut to match
    dimensions.  expansion=1 means the output channel count equals *planes*.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """Build one BasicBlock.

        Args:
            in_planes: Number of input channels.
            planes:    Number of output channels.
            stride:    Stride for the first convolution (2 to halve resolution).
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Identity shortcut when dimensions match; projection otherwise
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """Forward pass: residual path + shortcut, then ReLU."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """Generic ResNet backbone parameterised by block type and layer depths.

    Adapted for 32x32 input images (traffic sign crops) by using a 3x3 stem
    convolution with stride 1 instead of the original 7x7/stride-2 stem, and
    a 4x4 average-pool before the classifier to match the smaller feature maps.
    """

    def __init__(self, block, num_blocks, num_classes=5):  # 5 classes for traffic signs
        """Build the ResNet.

        Args:
            block:       Residual block class (e.g. BasicBlock).
            num_blocks:  List of 4 ints — number of blocks in each of the 4 stages.
            num_classes: Number of output classes for the final linear layer.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Stem: single conv layer suited to 32x32 images (no stride reduction here)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Four stages, each doubling channels; stages 2-4 halve spatial resolution
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Stack *num_blocks* residual blocks, applying *stride* only to the first.

        Args:
            block:      Residual block class.
            planes:     Output channels for this stage.
            num_blocks: How many blocks to stack.
            stride:     Stride for the first block (subsequent blocks use stride 1).

        Returns:
            nn.Sequential containing all blocks for this stage.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through all four stages, global average pool, then classifier."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Global average pool over the 4x4 feature maps produced by a 32x32 input
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=5):
    """Create a ResNet-18 model instance.

    Args:
        num_classes: Number of output classes.

    Returns:
        ResNet-18 model.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)