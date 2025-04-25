import torch
from torch import nn
import torch.nn.functional as F


class GraphConvolution_no_degree(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Conv1d(in_features, out_features, 1, bias=False)

    def forward(self, x, A):
        x = x.permute(0, 2, 1)
        x_hat = torch.bmm(A, x)
        x = self.fc(x_hat.permute(0, 2, 1))
        return x


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GraphConvolution_no_degree(3, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.gcn2 = GraphConvolution_no_degree(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.gcn3 = GraphConvolution_no_degree(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.gcn4 = GraphConvolution_no_degree(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.gcn5 = GraphConvolution_no_degree(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.gcn6 = GraphConvolution_no_degree(32, 16)

        self.linear = nn.Linear(32, 2)

    def forward(self, feat_xyz, adj=None, feat_sp=None):
        x = feat_xyz.transpose(1, 2)
        x1 = self.gcn1(x, adj)
        x1 = self.bn1(x1)
        x1 = torch.relu(x1)

        x2 = self.gcn2(x1, adj)
        x2 = self.bn2(x2)
        x2 = torch.relu(x2)

        x3 = self.gcn3(x2, adj)
        x3 = self.bn3(x3)
        x3 = torch.relu(x3)

        x4 = self.gcn4(x3, adj)
        x4 = self.bn4(x4)
        x4 = torch.relu(x4)

        x5 = self.gcn5(x4, adj)
        x5 = self.bn5(x5)
        x5 = torch.relu(x5)

        x6 = self.gcn6(x5, adj)

        x_1 = F.adaptive_max_pool1d(x6, 1).view(x.shape[0], -1)
        x_2 = F.adaptive_avg_pool1d(x6, 1).view(x.shape[0], -1)
        x = torch.cat((x_1, x_2), dim=1)

        x = self.linear(x)

        return x


if __name__ == '__main__':
    model = GCN()
    feat_xyz = torch.randn((1, 6, 3))
    adj = torch.randn((1, 6, 6))
    x = model(feat_xyz, adj)
    print(x.size())
