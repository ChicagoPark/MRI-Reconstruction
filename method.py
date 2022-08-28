import torch
from torch import nn

from network import DnCNN, UNet
from dataset import ftran, fmult


class CNNBlock(nn.Module):
    name = 'CNN'

    def __init__(self):
        super().__init__()

        # self.nn = DnCNN(
        #     dimension=2,
        #     depth=5,
        #     n_channels=32,
        #     i_nc=2,
        #     o_nc=2,
        # )

        self.nn = UNet(
            dimension=2,
            i_nc=2,
            o_nc=2,
            f_root=32,
            conv_times=3,
            is_bn=False,
            activation='relu',
            is_residual=False,
            up_down_times=3,
            is_spe_norm=True,
            padding=(0, 0)
        )

    def forward(self, x, P, S, y):
        #x_hat = torch.view_as_real(x).permute([0, 3, 1, 2])
        #x_hat = self.nn(x_hat)
        x_hat = self.nn(x)
        #x_hat = x_hat.permute([0, 2, 3, 1]).contiguous()
        #print(f"\n\nYoungil: x_hat.shape: {x_hat.shape}")
        #x_hat = torch.view_as_complex(x_hat)

        return x_hat

class ResBlock(nn.Module):
    def __init__(self, dimension, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        if dimension == 2:
            conv_fn = nn.Conv2d
            bn_fn = nn.BatchNorm2d
        elif dimension == 3:
            conv_fn = nn.Conv3d
            bn_fn = nn.BatchNorm3d
        else:
            raise ValueError()

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv_fn(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias))
            if bn:
                m.append(bn_fn(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

activation_fn = {
     'relu': lambda: nn.ReLU(),
    'lrelu': lambda: nn.LeakyReLU(0.2),
}

class EDSR(nn.Module):
    def __init__(self, dimension, n_resblocks, n_feats, res_scale, in_channels=1, out_channels=1, act='relu'):
        super().__init__()

        if dimension == 2:
            conv_fn = nn.Conv2d
        elif dimension == 3:
            conv_fn = nn.Conv3d
        else:
            raise ValueError()

        m_head = [conv_fn(in_channels, n_feats, 3, padding=3 // 2)]

        m_body = [
            ResBlock(
                dimension, n_feats, 3, res_scale=res_scale, act=activation_fn[act](),
            ) for _ in range(n_resblocks)
        ]

        m_body.append(conv_fn(n_feats, n_feats, 3, padding=3 // 2))

        m_tail = [
            conv_fn(n_feats, out_channels, 3, padding=3 // 2)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, P = None, S = None, y = None):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

import json

class DeepUnfoldingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = CNNBlock()
        self.gamma = 0.5
        self.alpha = 0.5
        '''
        with open('config.json') as File:
            config = json.load(File)

        self.nn = EDSR(
            n_resblocks=config['module']['recon']['EDSR']['n_resblocks'],
            n_feats=config['module']['recon']['EDSR']['n_feats'],
            res_scale=config['module']['recon']['EDSR']['res_scale'],
            in_channels=2,
            out_channels=2,
            dimension=2, )
        self.gamma = 0.1
        self.alpha = 1.0
        '''

    def forward(self, x, P, S, y):
        x = x.permute([0, 2, 3, 1]).contiguous()
        dc, S = fmult(x, S, P)  # A x

        #print(f"\n\ndc.shape: {dc.shape}\ny.shape: {y.shape}")

        dc = ftran(dc - y, S, P)  # A^H (Ax - y)

        x = x - self.gamma * dc  # x^+ = x - gamma * A^H (Ax - y)

        x = x.permute([0, 3, 1, 2]).contiguous()

        prior = self.nn(x, P, S, y)

        return self.alpha * prior + (1 - self.alpha) * x

class DeepUnfolding(nn.Module):
    name = 'DU'

    def __init__(self, iterations):
        super().__init__()

        self.du_block = DeepUnfoldingBlock()
        self.iterations = iterations

    def forward(self, x, P, S, y):

        for _ in range(self.iterations):
            x = self.du_block(x, P, S, y)

        return x


def anderson_solver(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-4, beta=1.0):
    """ Anderson's acceleration for fixed point iteration. """

    bsz, H, W = x0.shape
    X = torch.zeros(bsz, m, H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, H * W, dtype=x0.dtype, device=x0.device)

    # bsz, d, H, W = x0.shape
    # X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    # F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)

    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []

    iter_ = range(2, max_iter)

    for k in iter_:
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))

        if res[-1] < tol:
            break

    return X[:, k % m].view_as(x0), res


class DEQ(nn.Module):
    name = 'DEQ'

    def __init__(self):
        super().__init__()

        self.du_block = DeepUnfoldingBlock()

    def forward(self, x, P, S, y):

        with torch.no_grad():
            z_fixed, forward_res = anderson_solver(
                lambda z: self.du_block(z, P, S, y), x,
                max_iter=100,
                tol=1e-3,
            )

            forward_iter = len(forward_res)
            forward_res = forward_res[-1]

        x_hat = self.du_block(z_fixed, P, S, y)
        # JFB

        return x_hat, forward_iter, forward_res
