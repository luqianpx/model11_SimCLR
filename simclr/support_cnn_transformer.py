import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_padding(in_dim, out_dim, kernel_size, stride):
    """Helper function to calculate padding size."""
    p = max(0, (out_dim - 1) * stride + kernel_size - in_dim)
    pad_left = p // 2
    pad_right = p - pad_left
    return pad_left, pad_right


class MyConv1dPadSame(nn.Module):
    """Convolution with SAME padding."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, groups=groups)

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        pad_left, pad_right = compute_padding(in_dim, out_dim, self.kernel_size, self.stride)
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.conv(x)


class MyMaxPool1dPadSame(nn.Module):
    """MaxPool with SAME padding."""
    def __init__(self, kernel_size):
        super().__init__()
        self.max_pool = nn.MaxPool1d(kernel_size)

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + 1 - 1) // 1
        pad_left, pad_right = compute_padding(in_dim, out_dim, self.kernel_size, 1)
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.max_pool(x)


class BasicBlock(nn.Module):
    """ResNet-like BasicBlock."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn=True, use_do=True):
        super().__init__()
        self.conv1 = MyConv1dPadSame(in_channels, out_channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)

        self.conv2 = MyConv1dPadSame(out_channels, out_channels, kernel_size, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)

        self.max_pool = MyMaxPool1dPadSame(stride)
        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        identity = x
        x = self.bn1(x) if self.use_bn else x
        x = self.relu1(x)
        x = self.do1(x) if self.use_do else x
        x = self.conv1(x)

        x = self.bn2(x) if self.use_bn else x
        x = self.relu2(x)
        x = self.do2(x) if self.use_do else x
        x = self.conv2(x)

        if self.downsample:
            identity = self.max_pool(identity)

        if self.out_channels != x.shape[1]:
            identity = F.pad(identity, (0, self.out_channels - x.shape[1]), "constant", 0)

        return x + identity


class PositionalEncoding(nn.Module):
    """Positional Encoding for transformers."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CNNTransformer(nn.Module):
    """Model integrating CNN and Transformer."""
    def __init__(self, in_channels=16, fft=200, steps=20, dropout=0.2, nhead=8, emb_size=256, n_segments=5, num_layers=30):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            BasicBlock(in_channels, 32, kernel_size=7, stride=2, groups=1, downsample=True),
            BasicBlock(32, 64, kernel_size=7, stride=2, groups=2, downsample=True),
            BasicBlock(64, 128, kernel_size=7, stride=2, groups=2, downsample=True),
            BasicBlock(128, 256, kernel_size=7, stride=2, groups=2, downsample=True)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=nhead, dim_feedforward=emb_size, dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(emb_size)

    def forward(self, x):
        n_length = x.shape[2] // self.n_segments
        cnn_emb = [self.conv_blocks(x[:, :, idx * n_length: idx * n_length + n_length]).unsqueeze(1) for idx in range(self.n_segments)]
        x = torch.cat(cnn_emb, dim=1)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        return x.mean(dim=1)


if __name__ == "__main__":
    x = torch.randn(2, 16, 1000)
    model = CNNTransformer(in_channels=16, fft=200, steps=2)
    out = model(x)
    print(out.shape)
