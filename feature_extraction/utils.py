from pathlib import Path
from random import shuffle
from functools import partial


def split_trainval(data_root: str, train_ratio=0.8):
    train_root = Path(data_root) / "train"
    cates = sorted([d.stem for d in train_root.glob('*') if d.is_dir()])
    train_list, val_list = [], []
    for idx, cate in enumerate(cates):
        samples = [d for d in (train_root / cate).glob('*') if d.is_dir()]
        shuffle(samples)
        len_train = int(len(samples) * train_ratio)
        assert len_train > 0
        for _i, sample in enumerate(samples):
            if _i < len_train:
                train_list.append({'path': str(sample.absolute()),'label': idx})
            else:
                val_list.append({'path': str(sample.absolute()),'label': idx})
    return train_list, val_list


def res2tab(res: dict, n_palce=4):
    def dy_str(s, l):
        return  str(s) + ' '*(l-len(str(s)))
    min_size = 8
    k_str, v_str = '', ''
    for k, v in res.items():
        cur_len = max(min_size, len(k)+2)
        k_str += dy_str(f'{k}', cur_len) + '| '
        v_str += dy_str(f'{v:.4}', cur_len) + '| '
    return '\n'.join([k_str, v_str])


class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, mode='max', patience=20, threshold=1e-4, threshold_mode='rel'):
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self.is_converged = False
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self.best = self.mode_worse

    def is_improved(self):
        return self.num_bad_epochs == 0

    def step(self, metrics):
        if self.is_converged:
            raise ValueError
        current = metrics
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.is_converged = True

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon
        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)


################################### metric #######################################
import scipy.spatial
import numpy as np


def acc_score(y_true, y_pred, average="micro"):
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if average == "micro": 
        # overall
        return np.mean(y_true == y_pred)
    elif average == "macro":
        # average of each class
        cls_acc = []
        for cls_idx in np.unique(y_true):
            cls_acc.append(np.mean(y_pred[y_true==cls_idx]==cls_idx))
        return np.mean(np.array(cls_acc))
    else:
        raise NotImplementedError


def map_score(dist_mat, lbl_a, lbl_b, metric='cosine'):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        p = 0.0
        r = 0.0
        for j in range(n_b):
            if lbl_a[i] == lbl_b[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res.append(p/r)
        else:
            res.append(0)
    return np.mean(res)


def map_score(dist_mat, lbl_a, lbl_b):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        p = 0.0
        r = 0.0
        for j in range(n_b):
            if lbl_a[i] == lbl_b[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res.append(p/r)
        else:
            res.append(0)
    return np.mean(res)


def nn_score(dist_mat, lbl_a, lbl_b):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        if lbl_a[i] == lbl_b[order[0]]:
            res.append(1)
        else:
            res.append(0)
    return np.mean(res)


def ndcg_score(dist_mat, lbl_a, lbl_b, k=100):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, n_b + 2)))
        dcg = np.cumsum([1.0/np.log2(idx+2) if lbl_a[i] == lbl_b[item] else 0.0 for idx, item in enumerate(order)])
        ndcg = (dcg/idcg)[k-1]
        res.append(ndcg)
    return np.mean(res)


def anmrr_score(dist_mat, lbl_a, lbl_b):
    # NG: number of ground truth images (target images) per query (vector)
    n_a, n_b = dist_mat.shape
    lbl_a, lbl_b = np.array(lbl_a), np.array(lbl_b)
    NG = np.array([(lbl_a[i]==lbl_b).sum() for i in range(lbl_a.shape[0])])
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        cur_NG = NG[i]
        K = min(4*cur_NG, 2*NG.max())
        order = s_idx[i]
        ARR = np.sum([(idx+1)/cur_NG if lbl_a[i] == lbl_b[order[idx]] else (K+1)/cur_NG for idx in range(cur_NG)])
        MRR = ARR - 0.5*cur_NG - 0.5
        NMRR = MRR / (K - 0.5*cur_NG + 0.5)
        res.append(NMRR)
    return np.mean(res)


def pr(dist_mat, lbl_a, lbl_b, top_k=1e9):
    n_a, n_b = dist_mat.shape
    top_k = min(top_k, n_b)
    s_idx = dist_mat.argsort()
    ans = []
    for i in range(n_a):
        cur_pr = [0]*11
        order = s_idx[i][:top_k]
        p_list, r_list = [], []
        truth = (lbl_a[i] == lbl_b[order])
        r_seen, r_max = 0, truth.sum()
        for j in range(top_k):
            if truth[j]:
                r_seen += 1
                r_list.append(r_seen / r_max)
                p_list.append(r_seen / (j + 1))
        if r_seen != 0:
            for ii in range(len(p_list)):
                p_list[ii] = max(p_list[ii:])
            r_list, p_list = np.array(r_list), np.array(p_list)
            for idx, t in enumerate(np.arange(0., 1.1, 0.1)):
                if np.sum(r_list >= t) != 0:
                    cur_pr[idx] = np.max(p_list[r_list >= t])
        ans.append(cur_pr)
    return np.array(ans).mean(0).tolist()

if __name__ == "__main__":
    split_trainval('data/OS-MN40')
