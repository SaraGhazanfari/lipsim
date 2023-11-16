import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_inv(x):
  mask = x == 0
  x_inv = x**(-1)
  x_inv[mask] = 0
  return x_inv


class SDPBasedLipschitzConvLayer(nn.Module):

    def __init__(self, cin, inner_dim, kernel_size=3, stride=1):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else cin
        self.activation = nn.ReLU()

        self.padding = kernel_size // 2

        self.kernel = nn.Parameter(torch.randn(inner_dim, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(1, inner_dim, 1, 1))
        self.q = nn.Parameter(torch.randn(inner_dim))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound) # bias init

    def compute_t(self):
        ktk = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        ktk = torch.abs(ktk)
        q = torch.exp(self.q).reshape(-1, 1, 1, 1)
        q_inv = torch.exp(-self.q).reshape(-1, 1, 1, 1)
        t = (q_inv * ktk * q).sum((1, 2, 3))
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = self.compute_t()
        t = t.reshape(1, -1, 1, 1)
        res = F.conv2d(x, self.kernel, padding=1)
        res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.conv_transpose2d(res, self.kernel, padding=1)
        out = x - res
        return out


class SDPBasedLipschitzLinearLayer(nn.Module):

    def __init__(self, cin, inner_dim):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else cin
        self.activation = nn.ReLU()

        self.weight = nn.Parameter(torch.empty(inner_dim, cin))
        self.bias = nn.Parameter(torch.empty(1, inner_dim))
        self.q = nn.Parameter(torch.randn(inner_dim))

        nn.init.xavier_normal_(self.weight)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound) # bias init

    def compute_t(self):
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = torch.abs(torch.einsum('i,ik,kj,j -> ij', q_inv, self.weight, self.weight.T, q)).sum(1)
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = self.compute_t()
        res = F.linear(x, self.weight)
        res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.linear(res, self.weight.T)
        out = x - res
        return out


class SDPConvLin(nn.Module):

    def __init__(self, cin, cout, kernel_size=3):
        super(SDPConvLin, self).__init__()

        self.activation = nn.ReLU(inplace=False)

        self.kernel = nn.Parameter(torch.empty(cin, cout, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.randn(cin))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound) # bias init

    def forward(self, x):
        batch_size, cout, x_size, x_size = x.shape
        ktk = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        q = torch.exp(self.q).reshape(-1, 1, 1, 1)
        q_inv = torch.exp(-self.q).reshape(-1, 1, 1, 1)
        t = (q_inv * ktk * q).sum((1, 2, 3))
        t = safe_inv(torch.sqrt(t))
        x = t[None, :, None, None] * x
        out = F.conv_transpose2d(x, self.kernel, padding=1) + self.bias[None, :, None, None]
        return out


class SDPLin(nn.Module):

    def __init__(self, cin, cout, bias=True):
        super(SDPLin, self).__init__()

        self.weight = nn.Parameter(torch.empty(cout, cin))
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(cout))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else: self.bias = None
        self.q = nn.Parameter(torch.rand(cin))

    def forward(self, x):
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        wtw = self.weight.T @ self.weight
        t = torch.abs(q_inv * wtw * q).sum(1)
        t = torch.sqrt(safe_inv(t))
        W = self.weight * t
        out =  F.linear(x, W, self.bias)
        return out









class PaddingChannels(nn.Module):

  def __init__(self, ncout, ncin=3, mode="zero"):
    super(PaddingChannels, self).__init__()
    self.ncout = ncout
    self.ncin = ncin
    self.mode = mode

  def forward(self, x):
    if self.mode == "clone":
      return x.repeat(1, int(self.ncout / self.ncin), 1, 1) / np.sqrt(int(self.ncout / self.ncin))
    elif self.mode == "zero":
      bs, _, size1, size2 = x.shape
      out = torch.zeros(bs, self.ncout, size1, size2, device=x.device)
      out[:, :self.ncin] = x
      return out


class PoolingLinear(nn.Module):

  def __init__(self, ncin, ncout, agg="mean"):
    super(PoolingLinear, self).__init__()
    self.ncout = ncout
    self.ncin = ncin
    self.agg = agg

  def forward(self, x):
    if self.agg == "trunc":
      return x[:, :self.ncout]
    k = 1. * self.ncin / self.ncout
    out = x[:, :self.ncout * int(k)]
    out = out.view(x.shape[0], self.ncout, -1)
    if self.agg == "mean":
      out = np.sqrt(k) * out.mean(axis=2)
    elif self.agg == "max":
      out, _ = out.max(axis=2)
    return out


