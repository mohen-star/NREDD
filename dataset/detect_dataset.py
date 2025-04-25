import os
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.utils import loadswc, loadspe, bulid_subgraphs


class DetectDataset(Dataset):
    def __init__(self, datapath):
        super(DetectDataset, self).__init__()
        self.root_path = datapath
        # self.swc_path = os.path.join(datapath,'raw_data')
        self.swc_path = os.path.join(datapath, 'labeled_data')
        self.spe_path = os.path.join(datapath, 'SP_feature')
        # self.img_path = os.path.join(datapath,'tif')
        self.lab_path = os.path.join(datapath, 'label')
        self.file_list = os.listdir(self.swc_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        swc_name = self.file_list[item]
        # lab_name = swc_name.replace('.swc', '.txt')
        swc = loadswc(os.path.join(self.swc_path, swc_name))

        spe_feat = loadspe(os.path.join(self.spe_path, swc_name[:-4]))
        spe_feat = spe_feat / spe_feat.max()

        subtree_feat = bulid_subgraphs(swc)
        data = []
        for subdata in subtree_feat:
            subswc_feat, subadj, idx_list = subdata
            if len(idx_list) <= 2:
                continue
            subadj = torch.from_numpy(subadj).float()
            subswc_feat = torch.from_numpy(subswc_feat).float()
            sublab = torch.from_numpy(np.eye(2)[swc[idx_list[0], 1].astype(np.int8)]).float()
            subspe_feat = torch.from_numpy(spe_feat[idx_list]).float()
            data.append([subadj, subswc_feat, subspe_feat, sublab])
        return swc_name, data

if __name__ == '__main__':
    train_data_root = r'..\data\error_detect_data\train'
    train_dataset = DetectDataset(train_data_root)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    swc_name, data = train_dataset[0]
    adj, feat_xyz, feat_sp, label = data[5]
    print(swc_name)
    print(feat_xyz.shape)
    print(feat_xyz)
    print(feat_sp.shape)
    print(label.shape)
