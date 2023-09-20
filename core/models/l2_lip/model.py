import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from core.models.l2_lip.layers import PoolingLinear, PaddingChannels
from core.models.l2_lip.layers import SDPBasedLipschitzConvLayer, SDPBasedLipschitzLinearLayer


class NormalizedModel(nn.Module):

    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.normalize = Normalize(mean, std)

    def forward(self, x):
        return self.model(self.normalize(x))


class Projection(nn.Module):

  def __init__(self, n_classes):
    super(Projection, self).__init__()
    self.v = 1/np.sqrt(n_classes)

  def forward(self, x):
    return torch.clamp(x, -self.v, self.v)


class L2LipschitzNetwork(nn.Module):

    def __init__(self, config, n_classes):
        super(L2LipschitzNetwork, self).__init__()
        self.depth = config.depth
        self.num_channels = config.num_channels
        self.depth_linear = config.depth_linear
        self.n_features = config.n_features
        self.conv_size = config.conv_size
        self.n_classes = n_classes

        imsize = 224
        self.conv1 = PaddingChannels(self.num_channels, 3, "zero")

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
        self.last = PoolingLinear(in_channels, self.n_classes, agg="trunc")
        # self.projection = Projection(self.n_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        # x = self.projection(x)
        return x

