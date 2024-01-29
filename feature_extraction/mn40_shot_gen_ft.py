import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import time
import json
import hydra
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from rich.progress import track
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from datasets import VOMM_Shot_Data
from sklearn.metrics import f1_score
from models import MV_ResNet18 as Model


def setup_seed():
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logging.info(f"random seed: {seed}")

@torch.no_grad()
def extract(data_loader, net):
    net.eval()
    all_lbls, all_preds = [], []
    all_fts = []

    st = time.time()
    for lbl, sample in track(data_loader):
        sample = sample.cuda()
        out, ft = net(sample, global_ft=True)

        _, preds = torch.max(out, 1)
        all_fts.append(ft.detach().cpu().numpy())
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())

    all_fts = np.concatenate(all_fts, axis=0)
    res = {
        'f1_micro': f1_score(all_lbls, all_preds, average="micro"),
        'f1_macro': f1_score(all_lbls, all_preds, average="macro")
    }
    logging.info(' | '.join([f"{k}:{v:.6f}" for k, v in res.items()]))
    return all_fts

@hydra.main(config_path=".", config_name="mn40_shot")
def main(cfg: DictConfig):
    setup_seed()
    modalit_cfg = {
        'image': {'n_view': cfg.arch.n_view}
    }
    # predefined
    cfg.arch.batch_size *= (os.environ['CUDA_VISIBLE_DEVICES'].count(',')+1)
    logging.info(OmegaConf.to_yaml(cfg))
    # start
    # init train_loader and val_loader
    logging.info("Loader Initializing...\n")
    train_set, val_set, test_set = VOMM_Shot_Data(cfg.path.data_root, cfg.path.split, modalit_cfg)
    logging.info(f'train samples: {len(train_set)}')
    logging.info(f'val samples: {len(val_set)}')
    logging.info(f'test samples: {len(test_set)}')

    all_name = [*train_set.name_list, *val_set.name_list, *test_set.name_list]
    all_lbl = [*train_set.label_idx_list, *val_set.label_idx_list, *test_set.label_idx_list]
    all_lbl_name = [*train_set.label_name_list, *val_set.label_name_list, *test_set.label_name_list]
    train_idx = [1]*len(train_set) + [0]*len(val_set) + [0]*len(test_set)
    val_idx = [0]*len(train_set) + [1]*len(val_set) + [0]*len(test_set)
    test_idx = [0]*len(train_set) + [0]*len(val_set) + [1]*len(test_set)

    train_loader = DataLoader(train_set, batch_size=cfg.arch.batch_size, shuffle=False,
                                               num_workers=cfg.n_worker)
    val_loader = DataLoader(val_set, batch_size=cfg.arch.batch_size, shuffle=False,
                                             num_workers=cfg.n_worker)
    test_loader = DataLoader(test_set, batch_size=cfg.arch.batch_size, shuffle=False,
                                             num_workers=cfg.n_worker)
    net = Model(train_set.n_class, cfg.arch)
    net = net.cuda()
    net = nn.DataParallel(net)

    # load from file
    # best_state_dict = torch.load('/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40_shot_20/2022-05-02_16-18-48/ckpt.pth')
    best_state_dict = torch.load('/home2/fengyifan/code/OSR/Extract-Feature/cache/mn40_shot_10/2022-05-02_20-37-07/ckpt.pth')
    net.module.load_state_dict(best_state_dict['net'])

    # testing
    print('extract train...')
    train_fts = extract(train_loader, net)
    print('extract val...')
    val_fts = extract(val_loader, net)
    print('extract test...')
    test_fts = extract(test_loader, net)
    
    # format
    all_ft = np.concatenate((train_fts, val_fts, test_fts), axis=0)
    data = {
        'feature': all_ft,
        'label': np.array(all_lbl),
        'train_idx': np.array(train_idx),
        'val_idx': np.array(val_idx),
        'test_idx': np.array(test_idx),
        'name': all_name,
        'label_name': all_lbl_name,
    }
    np.save('mn40__mv__10t10v.npy', data)

if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    logging.info(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
