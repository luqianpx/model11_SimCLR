import torch.nn as nn
from .support_cnn_transformer import CNNTransformer
from .support_resnet1d import ResNet1D

class SimCLR_encoder(nn.Module):

    def __init__(self, args):
        super(SimCLR_encoder, self).__init__()

        if args.model_type == 'resnet1d':
            self.encoder = ResNet1D(in_channels=args.n_channel,
                                    base_filters=args.base_filters,
                                    kernel_size=args.kernel_size,
                                    stride=args.stride,
                                    groups=args.groups,
                                    n_block=args.n_block,
                                    downsample_gap=args.downsample_gap,
                                    increasefilter_gap=args.increasefilter_gap)
            self.n_features = int(args.base_filters*2**int((args.n_block-1)/args.increasefilter_gap))
        elif args.model_type == 'cnntransf':
            self.encoder = CNNTransformer(in_channels=args.n_channel,
                                          fft=args.fft,
                                          steps=args.steps,
                                          dropout=args.dropout,
                                          nhead=args.nhead,
                                          emb_size=args.emb_size,
                                          n_segments=args.n_segments,
                                          num_layers=args.num_layers)
            self.n_features = args.emb_size

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, int(self.n_features/2), bias=True),
            nn.ReLU(),
            nn.Linear(int(self.n_features/2), args.projection_dim, bias=True),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j