from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from lipsim.core.models.l2_lip.model import L2LipschitzNetwork


class LinearNormalized(nn.Linear):

    def __init__(self, cin, cout, bias=True):
        super(LinearNormalized, self).__init__(cin, cout, bias)
        nn.init.eye_(self.weight)

    def forward(self, x):
        self.Q = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, self.Q, self.bias)


class LipschitzClassification(nn.Module):

    def __init__(self, config, n_classes):
        super().__init__()
        self.backbone = L2LipschitzNetwork(config, 1792)
        checkpoint = torch.load(config.lipsim_ckpt)
        new_checkpoint = {}
        for k, v in checkpoint['model_state_dict'].items():
            new_checkpoint[k.replace('module.model.', '')] = v
        self.backbone.load_state_dict(new_checkpoint)
        for name, parameters in self.backbone.named_parameters():
          parameters.requires_grad = False
        if config.last_layer == 'lipschitz':
          self.linear = LinearNormalized(1792, n_classes)
        elif config.last_layer == 'dense':
          self.linear = nn.Linear(1792, n_classes)
        else:
          raise ValueError("Define last layer")

    def forward(self, x):
        x = self.backbone(x)
        return self.linear(x)


