import sys

sys.path.append(".")

import os
import time
import torch
import random
import numpy as np
import scipy.spatial
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from metric_tools import map_score


def setup_seed():
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f"random seed: {seed}")


def get_data(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    fts, label = data["feature"], np.array(data["label"])
    train_idx, query_idx, target_idx = (
        data["train_idx"].astype("bool"),
        data["query_idx"].astype("bool"),
        data["target_idx"].astype("bool"),
    )
    train_lbls, query_lbls, target_lbls = (
        label[train_idx],
        label[query_idx],
        label[target_idx],
    )

    fts = torch.from_numpy(fts).cuda()
    train_lbls = torch.tensor(train_lbls).long().squeeze().cuda()
    query_lbls = torch.tensor(query_lbls).long().squeeze().cuda()
    target_lbls = torch.tensor(target_lbls).long().squeeze().cuda()
    train_idx = torch.from_numpy(train_idx).cuda()
    query_idx = torch.from_numpy(query_idx).cuda()
    target_idx = torch.from_numpy(target_idx).cuda()
    return fts, train_lbls, train_idx, query_idx, target_idx, query_lbls, target_lbls


##################### AutoEncoder ##########################
class AutoEncoder(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_ch, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, hid_ch),
        )
        self.decorder = nn.Sequential(
            nn.Linear(hid_ch, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, in_ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        en_x = self.encoder(x)
        re_x = self.decorder(en_x)
        return en_x, re_x


def train_AE(fts, train_idx, query_idx, target_idx, query_lbls, target_lbls, emb_dim, max_epoch=50):
    dist_mat = scipy.spatial.distance.cdist(
        fts[query_idx].cpu().numpy(), fts[target_idx].cpu().numpy(), "cosine"
    )
    print(
        f"raw ft map: {map_score(dist_mat, query_lbls.cpu().numpy(), target_lbls.cpu().numpy())}"
    )

    x = fts[train_idx]
    Net = AutoEncoder(x.size(1), emb_dim).cuda()
    optimizer = optim.SGD(Net.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
    loss_fn = nn.MSELoss().cuda()
    for epoch in range(max_epoch):
        Net.train()
        optimizer.zero_grad()
        en_x, re_x = Net(x)
        loss = loss_fn(re_x, x)
        loss.backward()
        optimizer.step()
        print(f"AE -> [{epoch}/{max_epoch}] loss: {loss.item():.5f}")
    Net.eval()
    en_fts, _ = Net(fts)
    dist_mat = scipy.spatial.distance.cdist(
        en_fts[query_idx].detach().cpu().numpy(),
        en_fts[target_idx].detach().cpu().numpy(),
        "cosine",
    )
    print(
        f"new ft map: {map_score(dist_mat, query_lbls.cpu().numpy(), target_lbls.cpu().numpy())}"
    )
    return en_fts.detach().cpu().numpy()


def npy_append_ae_ft(ae_ft, file_path, new_file_path):
    data = np.load(file_path, allow_pickle=True).item()
    data[f"ae_{ae_ft.shape[1]}"] = ae_ft
    np.save(new_file_path, data)


def main():
    emb_dim = 512
    # dataset = 'esb'
    dataset = "esb" # esb, ntu, mn40, abo
    marker = "t2r8"
    # dataset = 'mn40-abo'
    # marker = 'ex'
    setup_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_path = Path(f"feature/{dataset}__level_all__{marker}__vox32_voxnet.npy")
    # data_path = Path(f"feature/{dataset}__2set_test2__vox32_voxnet.npy")
    # data_path = Path(f"feature/{dataset}__level_all__{marker}__vox32_voxnet_aug.npy")
    new_data_path = data_path.with_name(f"{data_path.stem}_ae{data_path.suffix}")
    print(f"Load Data from {data_path}")
    fts, train_lbls, train_idx, query_idx, target_idx, query_lbls, target_lbls = get_data(data_path)
    ae_fts = train_AE(
        fts, train_idx, query_idx, target_idx, query_lbls, target_lbls, emb_dim
    )

    print(f"train samples: {train_lbls.shape[0]}")
    print(f"query samples: {query_lbls.shape[0]}")
    print(f"target samples: {target_lbls.shape[0]}")

    npy_append_ae_ft(ae_fts, data_path, new_data_path)
    print(f"AE feature see file {new_data_path}")


if __name__ == "__main__":
    all_st = time.time()
    main()
    all_sec = time.time() - all_st
    print(
        f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes {all_sec%60:.2f}s!"
    )
