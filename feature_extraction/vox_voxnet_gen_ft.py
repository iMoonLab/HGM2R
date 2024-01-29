import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import time
import json
import hydra
import torch
import random
import logging
import numpy as np
import scipy.spatial
import torch.nn as nn
from pathlib import Path
from rich.progress import track
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from datasets import VOMM_OSR_Data
from sklearn.metrics import f1_score
from models import VoxNet as Model
from utils import map_score, pr

def setup_seed():
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logging.info(f"random seed: {seed}")

@torch.no_grad()
def extract_ft_lbl(data_loader, net):
    net.eval()
    all_lbls, all_fts = [], []
    for lbl, sample in track(data_loader):
        sample = sample.cuda()
        _, ft = net(sample, global_ft=True)
        all_lbls.append(lbl.detach().cpu().numpy())
        all_fts.append(ft.detach().cpu().numpy())
    all_lbls = np.concatenate(all_lbls, axis=0)
    all_fts = np.concatenate(all_fts, axis=0)
    return all_fts, all_lbls

@hydra.main(config_path=".", config_name="vox_voxnet")
def main(cfg: DictConfig):
    if cfg.mark.startswith('mn40__2set'):
        # best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40__2set_test2__vox32_voxnet/2022-11-01_09-38-32/ckpt.pth"
        best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40__2set_test1__vox32_voxnet/2022-11-01_09-53-50/ckpt.pth"
    elif cfg.mark.startswith('mn40>abo'):
        # best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40>abo__level_all__in__vox32_voxnet/2022-10-17_12-24-51/ckpt.pth"
        best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40>abo__level_all__ex__vox32_voxnet/2022-10-17_13-08-11/ckpt.pth"
    elif cfg.mark.startswith('esb'):
        best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/esb__level_all__t2r8__vox32_voxnet/2022-04-27_19-05-05/ckpt.pth"
    elif cfg.mark.startswith('ntu'):
        best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/ntu__level_all__t2r8__vox32_voxnet/2022-04-27_19-22-39/ckpt.pth"
    elif cfg.mark.startswith('mn40'):
        # best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40__level_all__t2r8__vox32_voxnet/2022-04-27_18-13-58/ckpt.pth"
        # best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40__level_all__t2r8__vox32_voxnet_aug/2022-10-19_10-18-43/ckpt.pth"
        best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40__level_all__t2r8__vox64_voxnet_aug/2022-10-20_09-31-21/ckpt.pth"
        # best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40__level_all__t2r8_rnd__vox32_voxnet/2022-10-11_15-37-06/ckpt.pth"
        # best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40__level_all__t4r6__vox32_voxnet/2022-10-11_15-42-01/ckpt.pth"
        # best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40__level_all__t4r6_rnd__vox32_voxnet/2022-10-11_15-42-22/ckpt.pth"
        # best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40__level_all__t6r4__vox32_voxnet/2022-10-11_15-42-40/ckpt.pth"
        # best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40__level_all__t6r4_rnd__vox32_voxnet/2022-10-11_15-43-34/ckpt.pth"
    elif cfg.mark.startswith('abo'):
        best_ckpt_path = "/home2/fengyifan/code/OSR/Extract-Feature/cache/abo__level_all__t2r8__vox32_voxnet/2022-10-16_15-57-12/ckpt.pth"
    setup_seed()
    modalit_cfg = {
        'voxel': {'d_vox': cfg.arch.d_vox}
    }
    # predefined
    cfg.arch.batch_size *= (os.environ['CUDA_VISIBLE_DEVICES'].count(',')+1)
    logging.info(OmegaConf.to_yaml(cfg))
    # start
    # init train_loader and val_loader
    logging.info("Loader Initializing...\n")
    train_set, query_set, target_set = VOMM_OSR_Data(cfg.path.data_root, cfg.path.split, modalit_cfg)
    # train_set, query_set, target_set = VOMM_OSR_Data(cfg.path.data_root, cfg.path.split, modalit_cfg, data_ret_root=cfg.path.data_ret_root)
    logging.info(f'train samples: {len(train_set)}')
    logging.info(f'query samples: {len(query_set)}')
    logging.info(f'target samples: {len(target_set)}')

    all_name = [*train_set.name_list, *query_set.name_list, *target_set.name_list]
    all_lbl = [*train_set.label_idx_list, *query_set.label_idx_list, *target_set.label_idx_list]
    all_lbl_name = [*train_set.label_name_list, *query_set.label_name_list, *target_set.label_name_list]
    train_idx = [1]*len(train_set) + [0]*len(query_set) + [0]*len(target_set)
    query_idx = [0]*len(train_set) + [1]*len(query_set) + [0]*len(target_set)
    target_idx = [0]*len(train_set) + [0]*len(query_set) + [1]*len(target_set)

    train_loader = DataLoader(train_set, batch_size=cfg.arch.batch_size, shuffle=False,
                                               num_workers=cfg.n_worker)
    query_loader = DataLoader(query_set, batch_size=cfg.arch.batch_size, shuffle=False,
                                             num_workers=cfg.n_worker)
    target_loader = DataLoader(target_set, batch_size=cfg.arch.batch_size, shuffle=False,
                                             num_workers=cfg.n_worker)
    net = Model(train_set.n_class)
    net = net.cuda()
    net = nn.DataParallel(net)

    # load from file
    logging.info(f"Loading model from {best_ckpt_path}")
    net.module.load_state_dict(torch.load(best_ckpt_path)['net'])

    # extracting
    print('extract train...')
    tr_fts, tr_lbls = extract_ft_lbl(train_loader, net)
    print('extract query...')
    qu_fts, qu_lbls = extract_ft_lbl(query_loader, net)
    print('extract target...')
    ta_fts, ta_lbls = extract_ft_lbl(target_loader, net)

    # valid map score
    dist_mat = scipy.spatial.distance.cdist(qu_fts, ta_fts, "cosine")
    map_s = map_score(dist_mat, qu_lbls, ta_lbls)
    logging.info(f"retrieve set map_score: {map_s:.6f}")
    pr_s = pr(dist_mat, qu_lbls, ta_lbls)
    logging.info(f"retrieval pr curve:")
    logging.info(pr_s)

    all_lbl_copy = np.concatenate((tr_lbls, qu_lbls, ta_lbls), axis=0)
    all_ft = np.concatenate((tr_fts, qu_fts, ta_fts), axis=0)
    assert all_lbl == all_lbl_copy.tolist()

    data = {
        'feature': all_ft,
        'label': np.array(all_lbl),
        'train_idx': np.array(train_idx),
        'query_idx': np.array(query_idx),
        'target_idx': np.array(target_idx),
        'name': all_name,
        'label_name': all_lbl_name
    }
    np.save(f"{cfg.uuid}.npy", data)


if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    logging.info(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
