import math
import torch
from torch import nn


class GCNLayer(nn.Module):  # GCN层
    def __init__(self, input_features, output_features, bias=False, nomal='BN', actf='relu'):
        super(GCNLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = nn.Parameter(torch.FloatTensor(input_features, output_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if nomal == 'BN':
            self.Norm = nn.BatchNorm1d(output_features, affine=False, track_running_stats=False)  #
        elif nomal == 'LN':
            self.Norm = nn.LayerNorm(output_features, elementwise_affine=False, bias=False)
        else:
            self.Norm = nn.Sequential()
        if actf == 'relu':
            self.actf = nn.ReLU(inplace=True)
        else:
            self.actf = nn.Sequential()

    def reset_parameters(self):  # 初始化参数
        std = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x, adj):
        # print(x.shape,self.weights.shape)
        support = torch.mm(x, self.weights)
        # print(x.shape, adj.shape)
        output = torch.spmm(adj, support)
        # if output.mean().isnan():
        #     print("output")
        #     print(x,self.weights)
        if self.bias is not None:
            return output + self.bias
        output = self.actf(self.Norm(output))
        return output


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_config = [
            (3, 16), (16, 32), (32, 48), (48, 64),
            (64, 80), (80, 96), (96, 112), (112, 128), (128, 144)
        ]
        gcn_class = GCNLayer

        self.gcn_layers = nn.ModuleList()
        for in_ch, out_ch in self.layers_config[:-1]:
            self.gcn_layers.append(gcn_class(in_ch, out_ch))

        self.out_layer = gcn_class(self.layers_config[-1][0], self.layers_config[-1][1], nomal='none', actf='none')

    def forward(self, x, adj, feat_sp=None):

        for i, gcn in enumerate(self.gcn_layers):
            x = gcn(x, adj)

        x = self.out_layer(x, adj)

        return x


class Decoder(nn.Module):
    def forward(self, z):
        adj = torch.mm(z, z.transpose(0, 1))
        return adj


class GAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = Encoder()
        self.decode = Decoder()

    def forward(self, feat_xyz, adj, feat_sp):
        z = self.encode(feat_xyz, adj, feat_sp)
        reconstruction = self.decode(z)
        return reconstruction


class VEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_config = [
            (3, 16), (16, 32), (32, 48), (48, 64),
            (64, 80), (80, 96), (96, 112), (112, 128), (128, 144)
        ]
        gcn_class = GCNLayer

        self.gcn_layers = nn.ModuleList()
        for in_ch, out_ch in self.layers_config[:-1]:
            self.gcn_layers.append(gcn_class(in_ch, out_ch))

        self.mean_layer = gcn_class(self.layers_config[-1][0], self.layers_config[-1][1], nomal='none', actf='none')
        self.std_layer = gcn_class(self.layers_config[-1][0], self.layers_config[-1][1], nomal='none', actf='none')

    def forward(self, x, adj, feat_sp=None):
        for i, gcn, in enumerate(self.gcn_layers):
            x = gcn(x, adj)
        x_mean = self.mean_layer(x, adj)
        x_std = self.std_layer(x, adj)
        return x_mean, x_std


class VGAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = VEncoder()
        self.decode = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, feat_xyz, adj, feat_sp):
        z_mean, z_logvar = self.encode(feat_xyz, adj, feat_sp)
        z = self.reparameterize(z_mean, z_logvar)
        reconstruction = self.decode(z)
        return reconstruction, z_mean, z_logvar
