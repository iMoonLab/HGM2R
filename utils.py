import torch
import numpy as np
import numpy as np
from functools import partial



def load_data(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    if file_path.endswith("_ae.npy"):
        fts, label = data["ae_512"], np.array(data["label"])
    else:
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


# ========================================
# Hypergraph Related Functions
def dist2H(dist: torch.Tensor, top_k):
    H = torch.zeros_like(dist).long()
    _, tk_idx = dist.topk(top_k, dim=1, largest=False)
    col_idx = torch.arange(tk_idx.size(0)).unsqueeze(1).repeat(1, top_k)
    row_idx, col_idx = tk_idx.view(-1), col_idx.view(-1)
    H[row_idx, col_idx] = 1
    return H


def ft2H(ft: torch.Tensor, top_k):
    d = torch.cdist(ft, ft)
    if isinstance(top_k, list):
        Hs = []
        for _k in top_k:
            Hs.append(dist2H(d, _k))
        return Hs
    else:
        return dist2H(d, top_k)


def ft2G(ft: torch.Tensor, top_k=50, sym=False):
    Hs = None
    H = ft2H(ft, top_k)
    if isinstance(top_k, list):
        H = torch.stack(H)
    norm_r = 1 / H.sum(dim=1, keepdim=True)
    norm_r[torch.isinf(norm_r)] = 0
    norm_c = 1 / H.sum(dim=0, keepdim=True)
    norm_c[torch.isinf(norm_c)] = 0
    G = torch.matmul((norm_r * H), (norm_c * H).T)
    return G


def gcn_ft2knn(fts, top_k=50):
    n = fts.size(0)
    A = torch.zeros((n, n))
    cdist = torch.cdist(fts, fts)
    _, tk_idx = cdist.topk(top_k, dim=1, largest=False)
    node_idx = torch.arange(tk_idx.size(0)).unsqueeze(1).repeat(1, top_k)
    A[tk_idx, node_idx] = 1
    A[node_idx, tk_idx] = 1
    norm_r = 1 / A.sum(dim=1, keepdim=True)
    norm_r[torch.isinf(norm_r)] = 0
    A = A * norm_r + torch.eye(n)
    return A.cuda()


class EarlyStopping:
    def __init__(self, mode="max", patience=20, threshold=1e-4, threshold_mode="rel"):
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self.is_converged = False
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
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
        if mode == "min" and threshold_mode == "rel":
            rel_epsilon = 1.0 - threshold
            return a < best * rel_epsilon
        elif mode == "min" and threshold_mode == "abs":
            return a < best - threshold
        elif mode == "max" and threshold_mode == "rel":
            rel_epsilon = threshold + 1.0
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = float("inf")
        else:  # mode == 'max':
            self.mode_worse = -float("inf")

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)
