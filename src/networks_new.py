import math
from collections import OrderedDict

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.nn import init

from src.utils import bmtm, bmtv, bmmt, bbmv
from src.lie_algebra import SO3

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv1d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

class DPC(nn.Module):
    def __init__(self, in_dim, c0, ks, dropout, ds, momentum):
        super(DPC, self).__init__()
        self.dpc = nn.Sequential(
        torch.nn.Conv1d(in_dim, in_dim, ks, dilation=ds, groups=in_dim, bias=False),
        torch.nn.BatchNorm1d(in_dim, momentum=momentum),
        torch.nn.GELU(),
        torch.nn.Conv1d(in_dim, c0, 1, groups=1, bias=False),
        torch.nn.BatchNorm1d(c0, momentum=momentum),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        )
    def forward(self, x):
        x = self.dpc(x)
        return x

class BaseNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # channel dimension
        c1 = 2 * c0
        c2 = 2 * c1
        c3 = 2 * c2
        # kernel dimension (odd number)
        k0 = ks[0]
        k1 = ks[1]
        k2 = ks[2]
        k3 = ks[3]
        # dilation dimension
        d0 = ds[0]
        d1 = ds[1]
        d2 = ds[2]
        # padding
        p0 = (k0 - 1) + d0 * (k1 - 1) + d0 * d1 * (k2 - 1) + d0 * d1 * d2 * (k3 - 1)
        # nets
        self.start_pad = torch.nn.ReplicationPad1d((p0, 0))
        self.dpc1 = DPC(in_dim, c0, k0, dropout, 1, momentum)
        self.dpc2 = DPC(c0, c1, k1, dropout, d0, momentum)
        self.dpc3 = DPC(c1, c2, k2, dropout, d0*d1, momentum)
        self.dpc4 = DPC(c2, c3, k3, dropout, d0*d1*d2, momentum)
        self.fin = torch.nn.Conv1d(c3, out_dim, 1, dilation=1)
        self.end_pad = torch.nn.ReplicationPad1d((0, 0))
        self.lka1 = LKA(c0)
        self.lka2 = LKA(c1)
        self.lka3 = LKA(c2)
        self.lka4 = LKA(c3)

        self.mean_u = torch.nn.Parameter(torch.zeros(6),
            requires_grad=False)
        self.std_u = torch.nn.Parameter(torch.ones(6),
            requires_grad=False)

    def forward(self, us):
        u = self.norm(us).transpose(1, 2)
        y = self.start_pad(u)
        y = self.dpc1(y)
        y = self.dpc2(y)
        y = self.dpc3(y)
        y = self.dpc4(y)
        y = self.lka4(y)
        y = self.fin(y)
        y = self.end_pad(y)

        return y

    def norm(self, us):
        return (us-self.mean_u)/self.std_u

    def set_normalized_factors(self, mean_u, std_u):
        self.mean_u = torch.nn.Parameter(mean_u.cuda(), requires_grad=False)
        self.std_u = torch.nn.Parameter(std_u.cuda(), requires_grad=False)


class GyroNet(BaseNet):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum,
        gyro_std):
        super().__init__(in_dim, out_dim, c0, dropout, ks, ds, momentum)
        gyro_std = torch.Tensor(gyro_std)
        self.gyro_std = torch.nn.Parameter(gyro_std, requires_grad=False)

        gyro_Rot = 0.05*torch.randn(3, 3).cuda()
        self.gyro_Rot = torch.nn.Parameter(gyro_Rot)
        self.Id3 = torch.eye(3).cuda()

    def forward(self, us):
        ys = super().forward(us)
        Rots = (self.Id3 + self.gyro_Rot).expand(us.shape[0], us.shape[1], 3, 3)
        Rot_us = bbmv(Rots, us[:, :, :3])
        return self.gyro_std*ys.transpose(1, 2) + Rot_us

