import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from model.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class PointNet(nn.Module):
    def __init__(self, k=2, normal_channel=False):
        super(PointNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512, affine=False, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(256, affine=False, track_running_stats=False)
        self.relu = nn.ReLU()

    def forward(self, x, adj=None, feat_sp=None):
        x = x.transpose(1, 2)
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        # print('x: ', x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == '__main__':
    model = PointNet()
    feat_xyz = torch.randn((1, 6, 3))
    x, _ = model(feat_xyz)
    print(x.size())

