import torch
import torchvision
import torch.nn as nn
import random
import torch.nn.functional as F
from torchvision.models import inception_v3#, Inception_V3_Weights


class MV_FeatureNet(nn.Module):
    def __init__(self, pretrained=False):
        super(MV_FeatureNet, self).__init__()
        self.base_model = torchvision.models.resnet18(pretrained=pretrained)
        self.feature_len = 512
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    def forward(self, x):
        # feature maps
        x = self.features(x)
        # flatten
        x = x.view(x.size(0), -1)
        return x

class MV_ResNet18(nn.Module):
    def __init__(self, n_class, arch):
        n_view, pretrained = arch.n_view, arch.pretrained
        super(MV_ResNet18, self).__init__()
        self.n_view = n_view
        self.ft_net = MV_FeatureNet(pretrained=pretrained)
        self.cls_net = nn.Linear(self.ft_net.feature_len, n_class)

    def forward(self, view_batch, global_ft=False, local_ft=False):
        assert view_batch.size(1) == self.n_view
        view_batch = view_batch.view(-1, view_batch.size(2), view_batch.size(3), view_batch.size(4))
        view_fts = self.ft_net(view_batch)
        local_view_fts = view_fts.view(-1, self.n_view, view_fts.size(-1))
        global_view_fts, _ = local_view_fts.max(dim=1)
        outputs = self.cls_net(global_view_fts)
        if global_ft and local_ft:
            return outputs, global_view_fts, local_view_fts
        elif global_ft:
            return outputs, global_view_fts
        elif local_ft:
            return outputs, local_view_fts
        else:
            return outputs


class GV_FeatureNet(nn.Module):
    def __init__(self, pretrained=True):
        super(GV_FeatureNet, self).__init__()
        
        # self.base_model = torchvision.models.googlenet(init_weights=True, aux_logits=False)
        # self.feature_len = 1024
        # self.base_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.base_model = inception_v3(init_weights=True, aux_logits=False)
        self.feature_len = 2048

        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    def forward(self, x):
        # feature maps
        x = self.features(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc
        # x = self.fc_features(x)
        return x


class GVCNN(nn.Module):
    def __init__(self, n_class, arch):
        n_view, pretrained = arch.n_view, arch.pretrained
        super(GVCNN, self).__init__()
        self.n_view = n_view
        self.ft_net = GV_FeatureNet(pretrained=pretrained)
        self.cls_net = nn.Linear(self.ft_net.feature_len, n_class)

    def forward(self, view_batch, global_ft=False, local_ft=False):
        assert view_batch.size(1) == self.n_view
        view_batch = view_batch.view(-1, view_batch.size(2), view_batch.size(3), view_batch.size(4))
        view_fts = self.ft_net(view_batch)
        local_view_fts = view_fts.view(-1, self.n_view, view_fts.size(-1))
        global_view_fts, _ = local_view_fts.max(dim=1)
        outputs = self.cls_net(global_view_fts)
        if global_ft and local_ft:
            return outputs, global_view_fts, local_view_fts
        elif global_ft:
            return outputs, global_view_fts
        elif local_ft:
            return outputs, local_view_fts
        else:
            return outputs