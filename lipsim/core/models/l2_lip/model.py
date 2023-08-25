import torch.nn as nn
from torchvision.transforms import Normalize
from lipsim.core.models.l2_lip.layers import LinearNormalized, PoolingLinear, PaddingChannels
from lipsim.core.models.l2_lip.layers import SDPBasedLipschitzConvLayer, SDPBasedLipschitzLinearLayer


class NormalizedModel(nn.Module):

    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.normalize = Normalize(mean, std)

    def forward(self, x):
        return self.model(self.normalize(x))


class L2LipschitzNetwork(nn.Module):

    def __init__(self, config, n_classes):
        super(L2LipschitzNetwork, self).__init__()
        self.depth = config.depth
        self.num_channels = config.num_channels
        self.depth_linear = config.depth_linear
        self.n_features = config.n_features
        self.conv_size = config.conv_size
        self.n_classes = n_classes

        if config.dataset == 'tiny-imagenet':
            imsize = 64
        elif config.dataset in ['cifar10', 'cifar100']:
            imsize = 32
        else:
            imsize = 224

        self.conv1 = PaddingChannels(self.num_channels, 3, "zero")

        layers = []
        block_conv = SDPBasedLipschitzConvLayer
        block_lin = SDPBasedLipschitzLinearLayer

        for _ in range(self.depth):  # config, input_size, cin, cout, kernel_size=3, epsilon=1e-6
            layers.append(
                block_conv(config=config, input_size=(1, self.num_channels, imsize, imsize), cin=self.num_channels,
                           cout=self.conv_size))

        layers.append(nn.AvgPool2d(4, divisor_override=4))
        layers.append(nn.AvgPool2d(4, divisor_override=4))
        self.stable_block = nn.Sequential(*layers)

        layers_linear = [nn.Flatten()]
        if config.dataset in ['cifar10', 'cifar100']:
            in_channels = self.num_channels * 8 * 8
        elif config.dataset == 'tiny-imagenet':
            in_channels = self.num_channels * 16 * 16
        else:
            in_channels = self.num_channels * 14 * 14
        for _ in range(self.depth_linear):
            layers_linear.append(block_lin(config, in_channels, self.n_features))

        if config.last_layer == 'pooling_linear':
            self.last_last = PoolingLinear(in_channels, self.n_classes, agg="trunc")
        elif config.last_layer == 'lln':
            self.last_last = LinearNormalized(in_channels, self.n_classes)
        else:
            raise ValueError("Last layer not recognized")

        self.layers_linear = nn.Sequential(*layers_linear)
        self.base = nn.Sequential(*[self.conv1, self.stable_block, self.layers_linear])

    def forward(self, x):
        x = self.base(x)
        return self.last_last(x)

