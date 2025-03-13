"""
resnet for 1-d signal data, pytorch version

Shenda Hong, Oct 2019
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv1dPadSame(nn.Module):
    """
    Extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups)

    def forward(self, x):
        # Compute pad size
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.conv.stride[0] - 1) // self.conv.stride[0]
        padding = max(0, (out_dim - 1) * self.conv.stride[0] + self.conv.kernel_size[0] - in_dim)
        pad_left = padding // 2
        pad_right = padding - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.conv(x)


class MyMaxPool1dPadSame(nn.Module):
    """
    Extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x):
        # Compute pad size
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.max_pool.stride - 1) // self.max_pool.stride
        padding = max(0, (out_dim - 1) * self.max_pool.stride + self.max_pool.kernel_size[0] - in_dim)
        pad_left = padding // 2
        pad_right = padding - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.max_pool(x)


class BasicBlock(nn.Module):
    """
    ResNet Basic Block with optional BatchNorm, Dropout, and downsampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do):
        super(BasicBlock, self).__init__()

        self.conv1 = MyConv1dPadSame(in_channels, out_channels, kernel_size, stride, groups)
        self.bn1 = nn.BatchNorm1d(out_channels) if use_bn else None
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(0.5) if use_do else nn.Identity()

        self.conv2 = MyConv1dPadSame(out_channels, out_channels, kernel_size, 1, groups)
        self.bn2 = nn.BatchNorm1d(out_channels) if use_bn else None
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(0.5) if use_do else nn.Identity()

        self.downsample = downsample
        self.max_pool = MyMaxPool1dPadSame(kernel_size=stride) if downsample else nn.Identity()

    def forward(self, x):
        identity = x

        # First convolution block
        out = self.conv1(x)
        if self.bn1:
            out = self.bn1(out)
        out = self.relu1(out)
        out = self.do1(out)

        # Second convolution block
        out = self.conv2(out)
        if self.bn2:
            out = self.bn2(out)
        out = self.relu2(out)
        out = self.do2(out)

        # Downsample if needed
        if self.downsample:
            identity = self.max_pool(identity)

        # Shortcut connection
        out += identity
        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True):
        super(ResNet1D, self).__init__()

        self.first_block_conv = MyConv1dPadSame(in_channels, base_filters, kernel_size, 1)
        self.first_block_bn = nn.BatchNorm1d(base_filters) if use_bn else None
        self.first_block_relu = nn.ReLU()

        self.basicblock_list = nn.ModuleList()
        for i_block in range(n_block):
            downsample = (i_block % downsample_gap == 1)
            in_channels = base_filters * 2**((i_block) // increasefilter_gap) if i_block != 0 else base_filters
            out_channels = in_channels * 2 if (i_block % increasefilter_gap == 0 and i_block != 0) else in_channels
            block = BasicBlock(in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do)
            self.basicblock_list.append(block)

        self.final_bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.first_block_conv(x)
        if self.first_block_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        for block in self.basicblock_list:
            out = block(out)

        if self.final_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        return out.mean(-1)


if __name__ == "__main__":
    x = torch.randn(2, 6, 1000)
    model = ResNet1D(in_channels=6, base_filters=32, kernel_size=10, stride=2, groups=2, n_block=20, increasefilter_gap=2)
    out = model(x)
    print(out.shape)
