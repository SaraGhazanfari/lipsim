from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from lipsim.core.models.dists.dists_model import DISTS
from lipsim.core.models.l2_lip.layers import PoolingLinear, PaddingChannels
from lipsim.core.models.l2_lip.layers import SDPBasedLipschitzConvLayer, SDPBasedLipschitzLinearLayer


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
        self.v = 1 / np.sqrt(n_classes)

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
        self.config = config

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

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x


class ClassificationLayer(nn.Module):
    def __init__(self, config, embed_dim, n_classes):
        super(ClassificationLayer, self).__init__()
        self.config = config
        self.finetuning_layer = SDPBasedLipschitzLinearLayer(embed_dim, n_classes)

    def forward(self, x):
        return self.finetuning_layer(x)


class LPIPSMetric:
    def __init__(self, lpips_metric):
        self.lpips_metric = lpips_metric

    def get_distance_between_images(self, img_ref, img_left, img_right, requires_grad=False,
                                    requires_normalization=False):
        dist_0 = self.lpips_metric(img_ref, img_left)
        dist_1 = self.lpips_metric(img_ref, img_right)
        if not requires_grad:
            dist_0 = dist_0.detach()
            dist_1 = dist_1.detach().squeeze()
        return dist_0.squeeze(), dist_1.squeeze(), None


class DISTSMetric:
    def __init__(self):
        self.dists_metric = DISTS().cuda()

    def get_distance_between_images(self, img_ref, img_left, img_right, requires_grad=False,
                                    requires_normalization=False):
        dist_0 = self.dists_metric(img_ref, img_left)
        dist_1 = self.dists_metric(img_ref, img_right)
        if not requires_grad:
            dist_0 = dist_0.detach()
            dist_1 = dist_1.detach()
        return dist_0, dist_1, None


class PerceptualMetric(nn.Module):
    def __init__(self, backbone, requires_bias=True, n_classes=1792, device='cuda'):
        super(PerceptualMetric, self).__init__()
        self.backbone = backbone
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.requires_bias = requires_bias
        self.bias = nn.Parameter(
            (1 / sqrt(n_classes)) * torch.ones(1, n_classes))

    def add_one_dim_to_embed(self, embed_ref):
        embed_ref = embed_ref + self.bias
        return torch.concat((embed_ref, torch.ones(embed_ref.shape[0], 1, device=embed_ref.device)), dim=1)

    def add_bias_to_embed(self, embed_ref):
        norm_ref = torch.norm(embed_ref, p=2, dim=(1))
        bias = torch.ones_like(embed_ref)
        bias[norm_ref > 1, :] = torch.zeros(embed_ref.shape[1], device=bias.device)
        bias = (2 / sqrt(embed_ref.shape[1])) * bias
        return embed_ref + bias

    def forward(self, img_ref, img_left, img_right, requires_grad=False,
                requires_normalization=False):

        embed_ref = self.get_embedding_per_image(img_ref, requires_grad, requires_normalization)
        embed_x0 = self.get_embedding_per_image(img_left, requires_grad, requires_normalization)
        embed_x1 = self.get_embedding_per_image(img_right, requires_grad, requires_normalization)

        bound = torch.norm(embed_x0 - embed_x1, p=2, dim=(1)).unsqueeze(1)
        dist_0 = 1 - self.cos_sim(embed_ref, embed_x0)
        dist_1 = 1 - self.cos_sim(embed_ref, embed_x1)
        return dist_0, dist_1, bound

    def get_embedding_per_image(self, img, requires_grad, requires_normalization):
        embed = self.backbone(img)

        if not requires_grad:
            embed = embed.detach()

        if self.requires_bias:
            embed = self.add_one_dim_to_embed(embed)

        if requires_normalization:
            norm_embed = torch.norm(embed, p=2, dim=(1)).unsqueeze(1)
            embed = embed / norm_embed

        return embed
