import sys

sys.path.append(".")

import os
import time
import torch
import random
import numpy as np
import scipy.spatial
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim

from models import CMAE, HGNN
from utils import load_data, ft2G, EarlyStopping
from metric_tools import acc_score, map_score, eval_all_metric


"""
    两阶段训练
    step 1 点云+体素+多视图 多模态自编码 + 取Mean
    step 2 朴素的单模态HGNN + 字典学习
"""


def setup_seed():
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f"random seed: {seed}")


def train_CMAE(ft_list, net, criterion, optimizer, epoch):
    n_m = len(ft_list)
    net.train()

    st = time.time()
    optimizer.zero_grad()
    xs, codes, re_xs, cre_xs = net(ft_list)
    loss_homo, rec_loss, cre_loss = None, None, None
    for _i in range(n_m):
        if loss_homo is None:
            rec_loss = criterion(xs[_i], re_xs[_i])
            cre_loss = criterion(xs[_i], cre_xs[_i])
        else:
            rec_loss += criterion(xs[_i], re_xs[_i])
            cre_loss += criterion(xs[_i], cre_xs[_i])
        for _j in range(_i + 1, n_m):
            if loss_homo is None:
                loss_homo = criterion(codes[_i], codes[_j])
            else:
                loss_homo += criterion(codes[_i], codes[_j])
    loss_homo, loss_br = (
        loss_homo / (n_m * (n_m - 1) / 2),
        rec_loss / n_m + cre_loss / n_m,
    )
    loss = 0.6 * loss_homo + 0.2 * loss_br
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


def train_GCN(
    fts1, A1, lbls, train_idx, net, ce_criterion, mse_criterion, optimizer, epoch
):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    outs, g_ft = net(fts1, A1)
    outs = outs[train_idx]
    loss_ce = ce_criterion(outs, lbls)
    loss = loss_ce
    loss.backward()
    optimizer.step()

    _, preds = torch.max(outs, 1)
    preds = preds.squeeze().detach().cpu().numpy().tolist()
    lbls = lbls.squeeze().detach().cpu().numpy().tolist()

    acc_mi = acc_score(lbls, preds, average="micro")
    acc_ma = acc_score(lbls, preds, average="macro")
    print(
        f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}, O-acc: {acc_mi:.5f}, M-acc: {acc_ma:.5f}"
    )
    return loss.item()


def train_HGNN(
    fts1, A1, lbls, train_idx, net, ce_criterion, mse_criterion, optimizer, epoch
):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    outs, outs_re, g_ft, g_ft_re, embs = net(fts1, A1)
    outs, outs_re = outs[train_idx], outs_re[train_idx]
    loss_ce = (ce_criterion(outs, lbls) + ce_criterion(outs_re, lbls)) / 2
    loss_mr = mse_criterion(g_ft[train_idx], g_ft_re[train_idx])
    loss = 0.1 * loss_ce + 0.9 * loss_mr
    loss.backward()
    optimizer.step()

    _, preds = torch.max(outs, 1)
    preds = preds.squeeze().detach().cpu().numpy().tolist()
    lbls = lbls.squeeze().detach().cpu().numpy().tolist()

    acc_mi = acc_score(lbls, preds, average="micro")
    acc_ma = acc_score(lbls, preds, average="macro")
    print(
        f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}, O-acc: {acc_mi:.5f}, M-acc: {acc_ma:.5f}"
    )
    return loss.item()


@torch.no_grad()
def retrieval(fts1, A1, query, target, query_lbls, target_lbls, net):
    net.eval()
    st = time.time()
    fts = net(fts1, A1, global_ft=True)
    print(f"Retrieval: Epoch Time: {time.time()-st:.5f}")
    query_fts = fts[query].squeeze().detach().cpu().numpy()
    target_fts = fts[target].squeeze().detach().cpu().numpy()
    dist_mat = scipy.spatial.distance.cdist(query_fts, target_fts, "cosine")
    map_s = map_score(dist_mat, query_lbls.cpu().numpy(), target_lbls.cpu().numpy())
    print(f"\t -> mAP: {map_s:.5f}")
    return map_s, map_s, fts


def save_mae_fts(ae_ft, file_path, new_file_path):
    data = np.load(file_path, allow_pickle=True).item()
    data[f"feature"] = ae_ft
    np.save(new_file_path, data)


def step1(path_prefix):
    pt_fts, train_lbls, train_idx, query_idx, target_idx, query_lbls, target_lbls = load_data(f"{path_prefix}__pt1024_pointnet.npy")
    # pt_fts, train_lbls, train_idx, query_idx, target_idx, query_lbls, target_lbls = get_data(f"{path_prefix}__pt1024_dgcnn.npy")
    vox_fts, train_lbls, train_idx, query_idx, target_idx, query_lbls, target_lbls = load_data(f"{path_prefix}__vox32_voxnet_ae.npy")
    # vox_fts, train_lbls, train_idx, query_idx, target_idx, query_lbls, target_lbls = get_data(f"{path_prefix}__vox32_voxnet_aug_ae.npy")
    mv_fts, train_lbls, train_idx, query_idx, target_idx, query_lbls, target_lbls = load_data(f"{path_prefix}__mv12_resnet18.npy")
    # mv_fts, train_lbls, train_idx, query_idx, target_idx, query_lbls, target_lbls = get_data(f"{path_prefix}__mv4_resnet18.npy")

    print(f"train samples: {train_lbls.shape[0]}")
    print(f"query samples: {query_lbls.shape[0]}")
    print(f"target samples: {target_lbls.shape[0]}")

    # cross-modal autoencoder
    print("Step 1:")
    print("Create cross-modal auto-encoder model")
    net = CMAE([pt_fts.size(1), vox_fts.size(1), mv_fts.size(1)])
    net = net.cuda()
    net = nn.DataParallel(net)

    optimizer = optim.SGD(net.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=1e-5
    )

    es = EarlyStopping(mode="max", patience=10, threshold=0.001)
    criterion = nn.MSELoss()
    criterion = criterion.cuda()

    best_res, best_fts = None, None
    for epoch in range(40):
        # train
        train_CMAE([pt_fts, vox_fts, mv_fts], net, criterion, optimizer, epoch)
        lr_scheduler.step()
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                net.eval()
                # st = time.time()
                fts = net([pt_fts, vox_fts, mv_fts], global_ft=True)
                print(f"Retrieval:")
                query_fts = fts[query_idx].squeeze().detach().cpu().numpy()
                target_fts = fts[target_idx].squeeze().detach().cpu().numpy()
                dist_mat = scipy.spatial.distance.cdist(query_fts, target_fts, "cosine")
                map_s = map_score(
                    dist_mat,
                    query_lbls.cpu().numpy(),
                    target_lbls.cpu().numpy(),
                )
                print(f"\t -> mAP: {map_s:.5f}")
            es.step(map_s)
            # save checkpoint
            if es.is_improved():
                print("saving model...")
                best_res = map_s
                best_fts = deepcopy(fts.cpu().numpy())
            if es.is_converged:
                break

    print("\n AE Train Finished!")
    print(f"Best result: {best_res}!")
    print("eval all metrics")
    query_idx, target_idx, query_lbls, target_lbls = (
        query_idx.cpu().numpy(),
        target_idx.cpu().numpy(),
        query_lbls.cpu().numpy(),
        target_lbls.cpu().numpy(),
    )
    eval_all_metric(best_fts[query_idx], best_fts[target_idx], query_lbls, target_lbls)

    # save
    save_mae_fts(
        best_fts,
        f"{path_prefix}__pt1024_pointnet.npy",
        f"{path_prefix}__pt_vox_mv_mae.npy",
    )


def step2(path_prefix, top_k):
    # load
    file_path = f"{path_prefix}__pt_vox_mv_mae.npy"
    # file_path = f"{path_prefix}__pt_vox_mv_mae_o.npy"
    # file_path = f"{path_prefix}__pt1024_pointnet.npy"
    # file_path = f"{path_prefix}__mv1_resnet18.npy"
    mae_fts, train_lbls, train_idx, query_idx, target_idx, query_lbls, target_lbls = load_data(file_path)
    n_class = train_lbls.max().item() + 1
    # train HGNN
    print("Step 2:")
    G = ft2G(mae_fts, top_k)
    net = HGNN(n_class, mae_fts.size(1))
    # G = gcn_ft2knn(mae_fts, top_k)
    # net = GCN(n_class, mae_fts.size(1))
    net = net.cuda()

    optimizer = optim.SGD(net.parameters(), 0.001, momentum=0.9)

    es = EarlyStopping(mode="max", patience=10, threshold=0.0001)
    ce_criterion = nn.CrossEntropyLoss().cuda()
    mse_criterion = nn.MSELoss().cuda()

    best_res, best_fts = None, None
    for epoch in range(120):
        # train
        train_HGNN(mae_fts, G, train_lbls, train_idx, net, ce_criterion, mse_criterion, optimizer, epoch )
        # train_GCN(mae_fts, G, train_lbls, train_idx, net, ce_criterion, mse_criterion, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_state, res, cur_fts = retrieval(
                    mae_fts, G, query_idx, target_idx, query_lbls, target_lbls, net
                )

            es.step(val_state)
            # save checkpoint
            if es.is_improved():
                print("saving model...")
                best_res = res
                best_fts = deepcopy(cur_fts.cpu().numpy())
            if es.is_converged:
                break

    print("\nTrain Finished!")
    print(f"Best result: {best_res}!")
    print("eval all metrics")
    query_idx, target_idx, query_lbls, target_lbls = (
        query_idx.cpu().numpy(),
        target_idx.cpu().numpy(),
        query_lbls.cpu().numpy(),
        target_lbls.cpu().numpy(),
    )
    eval_all_metric(best_fts[query_idx], best_fts[target_idx], query_lbls, target_lbls)


def main():
    # configure
    dataset = "esb" # esb, ntu, mn40, abo
    ## abo->mn40
    # dataset = 'abo-mn40'
    ## mn40->abo
    # dataset = 'mn40-abo'
    path_prefix = f"feature/{dataset}__level_all__t2r8"
    # path_prefix = f"feature/{dataset}__level_all__in"
    # path_prefix = f"feature/{dataset}__2set_test1"
    # path_prefix = f"feature/{dataset}__2set_test2"
    # path_prefix = f"feature/{dataset}__level_all__ex"
    # path_prefix = f"feature/{dataset}__level_all__nobg"
    # path_prefix = f"feature/{dataset}__level_all__real"
    if dataset == "esb":
        top_k = 12
    elif dataset == "ntu":
        top_k = 10
    elif dataset == "mn40":
        top_k = 50
    elif dataset == "abo":
        top_k = 50
    # cross dataset open-set retrieval
    # abo -> mn40
    elif dataset == "abo-mn40":
        top_k = 50
    # mn40 -> abo
    elif dataset == "mn40-abo":
        top_k = 50
    else:
        raise NotImplementedError
    setup_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # init train_loader and test loader
    print("Loader Initializing...\n")
    step1(path_prefix)
    step2(path_prefix, top_k)


if __name__ == "__main__":
    all_st = time.time()
    main()
    all_sec = time.time() - all_st
    print(
        f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes {all_sec%60:.2f}s!"
    )
