import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from datasets import VOMM_OSR_Data
from sklearn.metrics import f1_score
from models import VoxNet as Model
from utils import AverageMeter, map_score, EarlyStopping

def setup_seed():
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logging.info(f"random seed: {seed}")

def train(data_loader, net, criterion, optimizer, epoch):
    logging.info(f"Epoch {epoch}, Training...")
    net.train()
    loss_meter = AverageMeter()
    all_lbls, all_preds = [], []

    st = time.time()
    for i, (lbl, sample) in enumerate(data_loader):
        sample = sample.cuda()
        lbl = lbl.cuda()

        optimizer.zero_grad()
        out = net(sample)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(out, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        loss_meter.update(loss.item(), lbl.shape[0])
        logging.info(f"\t[{i}/{len(data_loader)}], Loss {loss.item():.4f}")

    f1_micro = f1_score(all_lbls, all_preds, average="micro")
    f1_macro = f1_score(all_lbls, all_preds, average="macro")
    logging.info(f"Time: {time.time()-st:.4f}s, Loss: {loss_meter.avg:4f}")
    logging.info(f"{f1_micro=:.5f}, {f1_macro=:.5f}")
    logging.info("This Epoch Done!\n")
    return loss_meter.avg

@torch.no_grad()
def extract_ft_lbl(data_loader, net):
    net.eval()
    all_lbls, all_fts = [], []
    for lbl, sample in data_loader:
        sample = sample.cuda()
        _, ft = net(sample, global_ft=True)
        all_lbls.append(lbl.detach().cpu().numpy())
        all_fts.append(ft.detach().cpu().numpy())
    all_lbls = np.concatenate(all_lbls, axis=0)
    all_fts = np.concatenate(all_fts, axis=0)
    return all_fts, all_lbls

def retrieve_it(query_loader, target_loader, net, epoch):
    logging.info(f"{epoch=}, retrieving...")
    st = time.time()
    q_fts, q_lbls = extract_ft_lbl(query_loader, net)
    t_fts, t_lbls = extract_ft_lbl(target_loader, net)
    dist_mat = scipy.spatial.distance.cdist(q_fts, t_fts, "cosine")
    res = {
        'map': map_score(dist_mat, q_lbls, t_lbls)
    }
    logging.info(f"Epoch: {epoch}, Time: {time.time()-st:.4f}s")
    logging.info(' | '.join([f"{k}:{v:.6f}" for k, v in res.items()]))
    logging.info("This Epoch Done!\n")
    return res['map'], res

def save_checkpoint(val_state, res, net: nn.Module):
    state_dict = net.state_dict()
    ckpt = dict(
        val_state=val_state,
        res=res,
        net=state_dict,
    )
    torch.save(ckpt, 'ckpt.pth')
    with open('ckpt.meta', 'w') as fp:
        json.dump(res, fp)

@hydra.main(config_path=".", config_name="vox_voxnet")
def main(cfg: DictConfig):
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
    
    train_loader = DataLoader(train_set, batch_size=cfg.arch.batch_size, shuffle=True,
                                               num_workers=cfg.n_worker, drop_last=True)
    query_loader = DataLoader(query_set, batch_size=cfg.arch.batch_size, shuffle=False,
                                             num_workers=cfg.n_worker)
    target_loader = DataLoader(target_set, batch_size=cfg.arch.batch_size, shuffle=False,
                                             num_workers=cfg.n_worker)
    logging.info("Create new model")
    net = Model(train_set.n_class)
    net = net.cuda()
    net = nn.DataParallel(net)

    optimizer = optim.SGD(net.parameters(), cfg.arch.lr, momentum=cfg.arch.momentum, \
        weight_decay=cfg.arch.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.arch.cos.T_max, 
                                                        eta_min=cfg.arch.cos.eta_min)

    es = EarlyStopping(mode="max", patience=cfg.arch.es.patience//cfg.arch.val_interval, 
                       threshold=cfg.arch.es.threshold)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    best_res = None
    for epoch in range(cfg.arch.max_epoch):
        # train
        train(train_loader, net, criterion, optimizer, epoch)
        lr_scheduler.step()
        # validation
        if epoch != 0 and epoch % cfg.arch.val_interval == 0:
            logging.info('retrieval...')
            ret_state, res = retrieve_it(query_loader, target_loader, net, epoch)
            # save checkpoint
            es.step(ret_state)
            if es.is_improved():
                logging.info("saving model...")
                best_res = res
                save_checkpoint(ret_state, res, net.module)
            if es.is_converged:
                break
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info("\nTrain Finished!")
    logging.info(' | '.join([f"{k}:{v:.6f}" for k, v in best_res.items()]))
    logging.info(f'checkpoint can be found in {Path.cwd()}!')
    return best_res

if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    logging.info(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
