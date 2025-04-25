import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distances = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distances.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=4, idx=None, dim4=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if not dim4:
            idx = knn(x, k)
        else:
            idx = knn(x[:, 0:3], k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch, num_dims, num_points)->(batch, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1,
                                                         2).contiguous()  # (batch_size, num_points, k, num_dims) -> (batch_size, num_dims*2, num_points, k)
    return feature


class DGCNN(nn.Module):
    def __init__(self, k=5, num_sp=9):
        super().__init__()

        self.k = k
        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm1d(1024)

        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(128)

        self.conv0 = nn.Sequential(nn.Conv2d(in_channels=3 * 2, out_channels=64, kernel_size=1, bias=False),
                                   self.bn0, nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(256 * 2, 1024, kernel_size=1, bias=False),
                                   self.bn4, nn.LeakyReLU(negative_slope=0.2))
        # 分类
        self.ln1 = nn.Linear(1024 * 2, 512, bias=False)
        self.dp1 = nn.Dropout(p=0.5)
        self.ln2 = nn.Linear(512, 256, bias=False)
        self.dp2 = nn.Dropout(p=0.5)
        self.ln3 = nn.Linear(256, 128, bias=False)
        self.dp3 = nn.Dropout(p=0.5)
        self.ln4 = nn.Linear(128, 2, bias=False)

    def forward(self, feat_xyz, adj=None, feat_sp=None):
        x = feat_xyz.transpose(2, 1)
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k, dim4=False)
        x = self.conv0(x)
        x0 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x0, k=self.k, dim4=False)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k, dim4=False)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k, dim4=False)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x0, x1, x2, x3), dim=1)

        x = self.conv4(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # 此处得到所需的特征(Batch, 1024+64*3, num_points)
        x = F.leaky_relu_(self.ln1(x))
        x = self.dp1(x)
        x = F.leaky_relu_(self.ln2(x))
        x = self.dp2(x)
        x = F.leaky_relu_(self.ln3(x))
        x = self.dp3(x)
        x = self.ln4(x)

        return x


if __name__ == '__main__':
    model = DGCNN()
    feat_xyz = np.random.random((1, 6, 3))
    feat_xyz = torch.from_numpy(feat_xyz).float()
    feat_tif = np.random.random((6, 9, 16, 16))
    feat_tif = torch.from_numpy(feat_tif).float()
    output = model(feat_xyz=feat_xyz, feat_sp=feat_tif)
    print(output.shape)
