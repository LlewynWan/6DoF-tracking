import torch
import torch.nn as nn

from typing import Type, Any, Callable, Union, List, Optional


def conv1x1(in_channel: int, out_channel: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channel: int, out_channel: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    kernel_size = np.asarray((3,3))
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    full_padding = (upsampled_kernel_size - 1) // 2
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)
    
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
            stride=stride, padding=full_padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channel: int, num_channel: int,
            stride: int = 1, dilation: int = 1,
            downsample: Optional[nn.Module] = None) -> None:
        
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channel, num_channel, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(num_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(num_channel, num_channel, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(num_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channel: int, num_channel: int,
            stride: int = 1, dilation: int = 1,
            downsample: Optional[nn.Module] = None):

        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channel, num_channel)
        self.bn1 = nn.BatchNorm2d(num_channel)

        self.conv2 = conv3x3(num_channel, num_channel, stride=stride, dilation=dilation)
        
        self.bn2 = nn.BatchNorm2d(num_channel)
        self.conv3 = conv1x1(num_channel, num_channel * 4)
        self.bn3 = nn.BatchNorm2d(num_channel * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
