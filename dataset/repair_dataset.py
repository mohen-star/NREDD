import os

import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.utils import load_data_for_connect, loadswc, load_SPfeature, get_edge_for_coonect, \
    get_edge_for_coonect_test, norm_feature_xyz


class RepairDataset(Dataset):
    def __init__(self, root: str, is_aug):
        self.is_aug = is_aug
        self.feat_xyz, self.adj, self.label, self.swc_path = load_data_for_connect(root, Ma=16,Mp=16)

    def __len__(self):
        return len(self.swc_path)

    def __getitem__(self, item):
        feat_xyz = self.feat_xyz[item]
        label = self.label[item]
        adj = self.adj[item]
        path = self.swc_path[item]
        feat_xyz = torch.from_numpy(feat_xyz).float()
        adj = torch.from_numpy(adj).float()
        label = torch.from_numpy(label).float()
        return feat_xyz, label, adj, path


class RepairDataset2(Dataset):
    def __init__(self, root: str, is_aug):
        self.swc_root = os.path.join(root, 'raw_data')
        self.label_root = os.path.join(root, 'label')
        self.spe_root = os.path.join(root, 'SP_feature')
        self.path_list = os.listdir(self.swc_root)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, item):
        swc_name = self.path_list[item]
        lab_name = swc_name.replace('.swc', '.txt')
        swc = loadswc(os.path.join(self.swc_root, swc_name))
        label = np.loadtxt(os.path.join(self.label_root, lab_name))
        feat_spe = load_SPfeature(os.path.join(self.swc_root, swc_name))
        feat_xyz = norm_feature_xyz(swc[:, 2:5])
        adj = get_edge_for_coonect_test(swc)

        # feat_spe = torch.from_numpy(feat_spe).float()
        feat_xyz = torch.from_numpy(feat_xyz).float()
        adj = torch.from_numpy(adj).float()
        label = torch.from_numpy(label).float()
        return feat_spe, feat_xyz, adj, label




if __name__ == '__main__':
    data_root = '../data/data_for_repair/test'
    dataset = RepairDataset(data_root, is_aug=False)
    # print(len(dataset))
    feat_xyz, label, adj, path = dataset[0]
    feat_sp = load_SPfeature(path)
    # print(feat_spe.shape)
    # print(feat_xyz.shape)
    # print(label.shape)
    # print(adj.shape)
