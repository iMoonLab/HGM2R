import torch
import torch.nn as nn
from collections import OrderedDict


class VoxNet(nn.Module):
    def __init__(self, n_classes):
        super(VoxNet, self).__init__()
        self.n_classes = n_classes
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))

        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(32 * 6 * 6 * 6, 128)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(128, self.n_classes))
        ]))

    def forward(self, x, global_ft=False):
        x = self.feat(x)
        g_ft = x.view(x.size(0), -1)
        x = self.mlp(g_ft)
        if global_ft:
            return x, g_ft
        else:
            return x


if __name__ == "__main__":
    voxnet = VoxNet(32, 10)
    data = torch.rand([256, 1, 32, 32, 32])
    voxnet(data)
