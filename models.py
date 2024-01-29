import torch
import torch.nn as nn
import torch.nn.functional as F


def L_B_R_D(dim_in, dim_out, drop_rate=0.5, with_bias=True):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out, bias=with_bias),
        nn.BatchNorm1d(dim_out),
        nn.ReLU(inplace=True),
        nn.Dropout(drop_rate),
    )


class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.trans = L_B_R_D(dim_in, dim_out)

    def forward(self, X, A):
        X = self.trans(X)
        X = torch.matmul(A, X)
        return X


class GCN(nn.Module):
    def __init__(self, n_class, m_in):
        super().__init__()
        self.m1_conv1 = GCNConv(m_in, 256)
        self.cls_layer = nn.Linear(256, n_class)

    def forward(self, X1, A1, global_ft=False):
        g_ft = self.m1_conv1(X1, A1)
        out = self.cls_layer(g_ft)
        if global_ft:
            return g_ft
        else:
            return out, g_ft


class HGNN(nn.Module):
    def __init__(self, n_class, m1_in):
        super().__init__()
        self.memory = nn.Embedding(128, 256)
        self.m1_conv1 = GCNConv(m1_in, 256)
        self.cls_layer = nn.Linear(256, n_class)

    def forward(self, X1, A1, global_ft=False):
        # for m1
        g_ft = self.m1_conv1(X1, A1)
        # fuse
        M = self.memory.weight
        M = M.expand(X1.size(0), -1, -1)
        g_ft = g_ft.unsqueeze(2)
        # ======================================================
        mem_key = F.softmax(torch.matmul(M, g_ft), dim=1)
        # ======================================================
        re_g_ft = mem_key * M
        re_g_ft = torch.mean(re_g_ft, dim=1)
        g_ft = g_ft.squeeze()
        out = self.cls_layer(g_ft)
        out_re = self.cls_layer(re_g_ft)
        if global_ft:
            return g_ft
        else:
            return out, out_re, g_ft, re_g_ft, self.memory.weight


def T(dim_in, dim_out):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.Tanh(),
    )


class CMAE(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.in_dims = in_dims
        self.heads = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for in_dim in in_dims:
            self.heads.append(T(in_dim, 1024))
            self.encoders.append(nn.Sequential(T(1024, 512)))
            self.decoders.append(nn.Sequential(T(512, 1024)))

    def forward(self, Xs, global_ft=False):
        xs = []
        for X, head in zip(Xs, self.heads):
            xs.append(head(X))
        # cs: common space
        codes = []
        for x, encoder in zip(xs, self.encoders):
            codes.append(encoder(x))
        # re: reconstruction
        re_xs = []
        for code, decoder in zip(codes, self.decoders):
            re_xs.append(decoder(code))
        # cr: cross-modal reconstruction
        cre_xs = []
        rand_idx = torch.randperm(len(self.in_dims), device=Xs[0].device)
        for _idx, decoder in enumerate(self.decoders):
            cre_xs.append(decoder(codes[rand_idx[_idx]]))
        if global_ft:
            g_ft = torch.stack(codes).mean(0)
            return g_ft
        else:
            return xs, codes, re_xs, cre_xs
