import json
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from pathlib import Path
from functools import partial
import torchvision.transforms as T
from typing import List, Union, Dict
from torch.utils.data import Dataset, DataLoader
# from torchdata.datapipes.iter import IterableWrapper, Mapper

def fetch_img_list(path: Union[str, Path], n_view, real=False):
    if not real:
        all_filenames = sorted(list(Path(path).glob('image/h_*.jpg')))
    else:
        all_filenames = sorted(list(Path(path).glob('real_image/main_*.jpg')))
    all_view = len(all_filenames)
    filenames = all_filenames[::all_view//n_view][:n_view]
    return filenames

def read_image(path_list: Union[List[str], List[Path]], augment=False, img_size=224):
    if augment:
        transform = T.Compose([
                T.RandomResizedCrop((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
    else:
        transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])
    imgs = [transform(Image.open(v).convert("RGB")) for v in path_list]
    imgs = torch.stack(imgs)
    return imgs

def fetch_pt_path(path: Union[str, Path], n_pt):
    p = Path(path) / 'pointcloud' / f'pt_{n_pt}.pts'
    if p.exists():
        return p
    else:
        return Path(path).with_suffix('.pts')

def read_pointcloud(path: Union[str, Path], augment=False):
    pt = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    pt = pt - np.expand_dims(np.mean(pt, axis=0), 0)  
    dist = np.max(np.sqrt(np.sum(pt ** 2, axis=1)), 0)
    pt = pt / dist  
    if augment:
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        pt = np.add(np.multiply(pt, xyz1), xyz2).astype('float32')
    pt = torch.from_numpy(pt.astype(np.float32))
    return pt.transpose(0, 1)

def fetch_vox_path(path: Union[str, Path], d_vox):
    return Path(path) / 'voxel' / f'vox_{d_vox}.ply'

def read_voxel(path: Union[str, Path], d_vox, augment=False):
    vox_3d = o3d.io.read_voxel_grid(str(path))
    vox_idx = torch.from_numpy(np.array([v.grid_index for v in vox_3d.get_voxels()])).float()
    vox_idx = vox_idx/vox_idx.max()
    # if False and augment:
    if False and augment:
        vox_idx = vox_idx * 2 - 1
        theta = torch.rand(1) * 2 * torch.pi
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                [torch.sin(theta), torch.cos(theta), 0],
                                [0, 0, 1]])
        vox_idx = torch.mm(vox_idx, rot_mat)
        vox_idx = (vox_idx - vox_idx.min(0)[0]) / 2
    vox_idx = vox_idx * d_vox
    vox_idx = torch.clamp(vox_idx, 0, d_vox - 1).long()
    vox = torch.zeros((d_vox, d_vox, d_vox))
    vox[vox_idx[:,0], vox_idx[:,1], vox_idx[:,2]] = 1
    return vox.unsqueeze(0)

class VOMM_dataset(Dataset):
    def __init__(self, data_root, sample_list, modality_cfg, augment=False, real=False):
        super().__init__()
        data_root = Path(data_root)
        self.real=real
        self.augment = augment
        self.cfg = modality_cfg
        self.path_list = [data_root / sample['path'] for sample in sample_list]
        self.label_idx_list = [sample['label_idx'] for sample in sample_list]
        self.name_list = [s['path'] for s in sample_list]
        self.label_name_list = [s['label'] for s in sample_list]
        self.n_class = len(set(self.label_idx_list))
    
    def __getitem__(self, index):
        cur_path = self.path_list[index]
        ret = [self.label_idx_list[index], ]
        for m, m_cfg in self.cfg.items():
            if m == 'image':
                img_list = fetch_img_list(cur_path, **m_cfg, real=self.real)
                data = read_image(img_list, augment=self.augment)
            elif m == 'pointcloud':
                path = fetch_pt_path(cur_path, **m_cfg)
                data = read_pointcloud(path, augment=self.augment)
            elif m == 'voxel':
                path = fetch_vox_path(cur_path, **m_cfg)
                data = read_voxel(path, **m_cfg, augment=self.augment)
            ret.append(data)
        return ret
    
    def __len__(self):
        return len(self.path_list)

def VOMM_Shot_Data(data_root, split_file, modality_cfg):
    with open(split_file, 'r') as f:
        split = json.load(f)
    train_set = VOMM_dataset(data_root, split['train'], modality_cfg, augment=True)
    val_set = VOMM_dataset(data_root, split['validation'], modality_cfg, augment=False)
    test_set = VOMM_dataset(data_root, split['test'], modality_cfg, augment=False)
    return train_set, val_set, test_set

def VOMM_OSR_Data(data_root, split_file, modality_cfg, data_ret_root=None, train_real=False, real=False):
    with open(split_file, 'r') as f:
        split = json.load(f)
    if data_ret_root is None:
        data_ret_root = data_root
    if train_real:
        train_data = VOMM_dataset(data_root, split['train']['sample'], modality_cfg, augment=True, real=True)
    else:
        train_data = VOMM_dataset(data_root, split['train']['sample'], modality_cfg, augment=True)
    query_data = VOMM_dataset(data_ret_root, split['retrieval']['query'], modality_cfg, augment=False, real=real)
    target_data = VOMM_dataset(data_ret_root, split['retrieval']['target'], modality_cfg, augment=False, real=real)
    return train_data, query_data, target_data

# def VOMM_datapipe(data_root: Union[str, Path], sample_list: List, modality_cfg: Dict[str, int], augment=False):
#     data_root = Path(data_root)
#     if isinstance(modality_cfg, str):
#         modality_cfg = {modality_cfg: None}
#     for m in modality_cfg.keys():
#         if m not in ['image', 'pointcloud', 'voxel']:
#             raise ValueError(f'Unknown modality: {m}')
    
#     path_list = [data_root / sample['path'] for sample in sample_list]
#     lbl_idx_list = [sample['label_idx'] for sample in sample_list]

#     # for different modality
#     data_dp_list = []
#     for m, m_cfg in modality_cfg.items():
#         if m == 'image':
#             cur_dp = IterableWrapper(path_list)
#             _fetch_img_list = partial(fetch_img_list, **m_cfg)
#             cur_dp = Mapper(cur_dp, _fetch_img_list)
#             _read_img = partial(read_image, augment=augment)
#             cur_dp = Mapper(cur_dp, _read_img)
#         elif m == 'pointcloud':
#             cur_dp = IterableWrapper(path_list)
#             _fetch_pt_path = partial(fetch_pt_path, **m_cfg)
#             cur_dp = Mapper(cur_dp, _fetch_pt_path)
#             _read_pt = partial(read_pointcloud, augment=augment)
#             cur_dp = Mapper(cur_dp, _read_pt)
#         elif m == 'voxel':
#             cur_dp = IterableWrapper(path_list)
#             _fetch_vox_path = partial(fetch_vox_path, **m_cfg)
#             cur_dp = Mapper(cur_dp, _fetch_vox_path)
#             _read_vox = partial(read_voxel, augment=augment)
#             cur_dp = Mapper(cur_dp, _read_vox)
#         data_dp_list.append(cur_dp)
#     # for label
#     dp_lbl = IterableWrapper(lbl_idx_list)
#     return dp_lbl.zip(*data_dp_list)

# def _get_datapipe(data_root, sample_list, modality_cfg, label_map, augment=False, detail=False):
#     cur_data = VOMM_datapipe(data_root, sample_list, modality_cfg, augment=augment)
#     cur_dataset = {'data': cur_data, 'n_class': len(label_map)}
#     if detail:
#         cur_dataset['name'] = [s['path'] for s in sample_list]
#         cur_dataset['label_name'] = [s['label'] for s in sample_list]
#     return cur_dataset

# def VOMM_Shot_Data(data_root, split_file, modality_cfg, detail=False):
#     with open(split_file, 'r') as f:
#         split = json.load(f)
#     train_set = _get_datapipe(data_root, split['train'], modality_cfg, split['label_map'], augment=False, detail=detail)
#     val_set = _get_datapipe(data_root, split['validation'], modality_cfg, split['label_map'], augment=False, detail=detail)
#     test_set = _get_datapipe(data_root, split['test'], modality_cfg, split['label_map'], augment=False, detail=detail)
#     return train_set, val_set, test_set


if __name__ == '__main__':
    data_root = '/home2/fengyifan/data/modelnet/40/ModelNet40_MM'
    osr_json_path = '/home2/fengyifan/code/OSR/Extract-Feature/splits/mn40__level_all__t2r8.json'
    shot_json_path = '/home2/fengyifan/code/OSR/Extract-Feature/splits/mn40__level_0__t20v20.json'
    modality_cfg = {
        'image': {'n_view':8},
        # 'pointcloud': {'n_pt': 1024},
        # 'voxel': {'n_vox': 32}
    }
    # img_root = Path('/media/fengyifan/本地磁盘/NTU/NTU_2000_MM/chess/Y3813_pawn/image')
    # img_list = sorted([str(p) for p in img_root.glob('*.jpg')])[::4]
    # imgs = read_image(img_list, augment=True)
    # print(imgs.shape)

    # pt = read_pointcloud('/media/fengyifan/本地磁盘/NTU/NTU_2000_MM/chess/Y3813_pawn/pointcloud/pt_1024.pts')
    # print(pt.shape)

    # vox = read_voxel('/media/fengyifan/本地磁盘/NTU/NTU_2000_MM/chess/Y3813_pawn/voxel/vox_32.ply')
    # print(vox.shape)

    # import json
    # with open(json_path, 'r') as fp:
    #     data = json.load(fp)
    # dp = VOMM_datapipe(data_root, data['train']['sample'], modality_cfg, augment=True)
    # batch = next(iter(dp))
    # print(batch[1].shape)

    train_set, val_set, test_set = VOMM_Shot_Data(data_root, shot_json_path, modality_cfg)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    for idx, (lbl, sample) in enumerate(train_dataloader):
        print(lbl)
