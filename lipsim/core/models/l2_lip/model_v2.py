import torch
import torch.nn as nn
from lipsim.core.models.l2_lip.layers import SDPBasedLipschitzConvLayer, SDPBasedLipschitzLinearLayer
from lipsim.core.models.l2_lip.layers import SDPConvLin, SDPLin


class L2LipschitzNetworkV2(nn.Module):

    def __init__(self, config, n_classes):
        super(L2LipschitzNetworkV2, self).__init__()
        self.depth = config.depth
        self.num_channels = config.num_channels
        self.depth_linear = config.depth_linear
        self.n_features = config.n_features
        self.conv_size = config.conv_size
        self.n_classes = n_classes
        self.config = config

        imsize = 224

        # self.conv1 = PaddingChannels(self.num_channels, 3, "zero")
        self.conv1 = SDPConvLin(3, self.num_channels)
        layers = []
        for _ in range(self.depth):  # config, input_size, cin, cout, kernel_size=3, epsilon=1e-6
            layers.append(
                SDPBasedLipschitzConvLayer(self.num_channels, self.conv_size)
            )
        layers.append(nn.AvgPool2d(4, divisor_override=4))
        layers.append(nn.AvgPool2d(4, divisor_override=4))
        self.convs = nn.Sequential(*layers)

        layers_linear = [nn.Flatten()]
        in_channels = self.num_channels * 14 * 14
        for _ in range(self.depth_linear):
            layers_linear.append(
                SDPBasedLipschitzLinearLayer(in_channels, self.n_features)
            )
        self.linear = nn.Sequential(*layers_linear)
        self.base = nn.Sequential(*[self.conv1, self.convs, self.linear])
        # self.last = PoolingLinear(in_channels, self.n_classes, agg="trunc")
        self.last = SDPLin(in_channels, self.n_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        if self.config.mode == 'ssa':
            x = x / torch.norm(x, p=2, dim=(1)).unsqueeze(1)
        return x
