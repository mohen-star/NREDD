import math

import torch
from thop import profile
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


class Encoder_IFE(nn.Module):
    def __init__(self, block='res', att=False, a_dp=False):
        super().__init__()
        self.att = att
        self.a_dp = a_dp
        self.layer_input = nn.Sequential(nn.Linear(3, 64),
                                         nn.BatchNorm1d(64, affine=False, track_running_stats=False),
                                         nn.ReLU())
        self.layer_sep = spe_feat_extract16(dropout=None)

        self.layers_config = [
            (192, 32), (32, 48), (48, 64),
            (64, 80), (80, 96), (96, 112), (112, 128), (128, 144)
        ]
        gcn_class = GCN_RESBLOCK

        self.gcn_layers = nn.ModuleList()
        self.att = nn.ModuleList()
        self.dp = nn.ModuleList()
        for in_ch, out_ch in self.layers_config[:-1]:
            self.gcn_layers.append(gcn_class(in_ch, out_ch))
            self.att.append(self_att_layer(out_ch))
            self.dp.append(AdvancedDropout(out_ch))
            # self.dp.append(nn.Dropout(0.5))

        self.out_layer = gcn_class(self.layers_config[-1][0], self.layers_config[-1][1], actf='none')

    def forward(self, feat_xyz, adj, feat_sp):
        # print(feat_sp.shape)
        swc_feat0 = self.layer_input(feat_xyz)
        spe_feat0 = self.layer_sep(feat_sp)
        spe_feat0 = spe_feat0.squeeze(dim=0)
        x = torch.cat((swc_feat0, spe_feat0), dim=1)
        # print(x.shape)
        for i, (gcn, att, dp) in enumerate(zip(self.gcn_layers, self.att, self.dp)):
            x = gcn(adj, x)
            if self.att and self.a_dp:
                x_a = att(x)
                x = dp(x + x_a)

        x = self.out_layer(adj, x)
        return x


class Decoder(nn.Module):
    def forward(self, z):
        adj = torch.mm(z, z.transpose(1, 0))
        return adj


class TLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = Encoder_IFE(att=True, a_dp=True)
        self.decode = Decoder()

    def forward(self, feat_xyz, adj, feat_sp):
        z = self.encode(feat_xyz, adj, feat_sp)
        reconstruction = self.decode(z)
        return reconstruction


class CONV2D(nn.Module):
    def __init__(self, input_features, output_features, nomal='BN', actf='relu', k=3, s=1, p=1, d=1):
        super(CONV2D, self).__init__()
        model = []
        model += [nn.Conv2d(input_features, output_features, kernel_size=k, stride=s, padding=p,
                            dilation=d)]  # padding_mode='circular'
        if nomal == 'BN':
            model += [nn.BatchNorm2d(output_features, affine=False, track_running_stats=False)]

        if actf == 'relu':
            model += [nn.ReLU(inplace=True)]
        elif actf == 'sigmoid':
            model += [nn.Sigmoid()]
        self.layer = nn.Sequential(*model)

    def forward(self, x):
        return self.layer(x)


class spe_feat_extract16(nn.Module):
    def __init__(self, dropout=0.5):
        super(spe_feat_extract16, self).__init__()
        self.conv1 = CONV2D(9, 32, k=3, s=2, p=1, d=1)
        # self.pool1 = nn.MaxPool2d(2,2)   #[9,16,16]->[32,8,8]
        self.conv2 = CONV2D(32, 32, k=3, s=1, p=1, d=1)
        # self.pool2 = nn.MaxPool2d(2, 2)  # [32,8,8]->[32,8,8]
        self.conv3 = CONV2D(32, 32, k=3, s=2, p=1, d=1)
        # self.pool3 = nn.MaxPool2d(2, 2)  # [32,8,8]->[32,4,4]
        self.conv4 = CONV2D(32, 32, k=3, s=1, p=1, d=2)
        # self.pool4 = nn.MaxPool2d(2, 2)  # [32,4,4]->[32,2,2]
        # self.conv5 = CONV2D(32,32)#,actf='sigmoid'
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

    def forward(self, x):
        y1 = self.dropout(self.conv1(x))
        # y1 = self.pool1(y1)
        # print("y1: ", y1.shape)
        y2 = self.dropout(self.conv2(y1))
        # y2 = self.pool2(y2)
        # print("y2: ", y2.shape)
        y3 = self.dropout(self.conv3(y2))
        # y3 = self.pool3(y3)
        # print("y3: ", y3.shape)
        y4 = self.dropout(self.conv4(y3))
        # y4 = self.pool4(y4)
        # print("y4: ", y4.shape)
        # out= self.conv5(y4)
        out = y4.view(1, x.shape[0], -1)
        # out = torch.softmax(out, dim=1)
        return out


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

    def forward(self, adj, x):
        # print(x.shape,self.weights.shape)
        support = torch.matmul(x, self.weights)
        output = torch.matmul(adj, support)
        # if output.mean().isnan():
        #     print("output")
        #     print(x,self.weights)
        if self.bias is not None:
            return output + self.bias
        output = self.actf(self.Norm(output))
        return output


class GCN_RESBLOCK(nn.Module):
    def __init__(self, input_size, output_size, bias=False, norm='BN', actf='relu'):
        super(GCN_RESBLOCK, self).__init__()
        self.GCNlayer = GCNLayer(input_size, output_size, bias=bias, nomal=norm)
        self.LinearLayer1 = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.BatchNorm1d(output_size, affine=False, track_running_stats=False))
        self.LinearLayer2 = nn.Sequential(nn.Linear(input_size, output_size),
                                          nn.BatchNorm1d(output_size, affine=False, track_running_stats=False))
        if actf == 'relu':
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.Sequential()

    def forward(self, adj, x):
        y1 = self.GCNlayer(adj, x)
        y1 = self.LinearLayer1(y1)

        y2 = self.LinearLayer2(x)
        return self.relu(y1 + y2)


class self_att_layer(nn.Module):
    def __init__(self, features):
        super(self_att_layer, self).__init__()
        self.k_layer = nn.Linear(features, features, bias=False)
        self.q_layer = nn.Linear(features, features, bias=False)
        self.v_layer = nn.Linear(features, features, bias=False)
        self.input_size = features

    def forward(self, x):
        # print(x.shape)
        k = self.k_layer(x)
        q = self.q_layer(x)
        v = self.v_layer(x)

        att_scores = torch.matmul(q, k.T) / torch.sqrt(torch.tensor(float(self.input_size)))
        att_weight = F.softmax(att_scores, dim=-1)
        output = torch.matmul(att_weight, v)
        # print(att_weight.shape,output.shape,v.shape)
        # if att_weight.mean().isnan():
        #     print(att_weight,output,v)
        return output


class AdvancedDropout(nn.Module):

    def __init__(self, num, init_mu=0, init_sigma=1.2, reduction=16):
        '''
        params:
        num (int): node number
        init_mu (float): intial mu
        init_sigma (float): initial sigma
        reduction (int, power of two): reduction of dimention of hidden states h
        '''
        super(AdvancedDropout, self).__init__()
        if init_sigma <= 0:
            raise ValueError("Sigma has to be larger than 0, but got init_sigma=" + str(init_sigma))
        self.init_mu = init_mu
        self.init_sigma = init_sigma

        self.weight_h = Parameter(torch.rand([num // reduction, num]).mul(0.01))
        self.bias_h = Parameter(torch.rand([1]).mul(0.01))

        self.weight_mu = Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_mu = Parameter(torch.Tensor([self.init_mu]))
        self.weight_sigma = Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_sigma = Parameter(torch.Tensor([self.init_sigma]))

    def forward(self, input):
        # print(self.training)
        if self.training:
            if len(input.size()) == 3:
                input1 = input.squeeze(dim=0)
            # print('training')
            c, n = input1.size()
            # parameterized prior
            h = F.linear(input, self.weight_h, self.bias_h)
            mu = F.linear(h, self.weight_mu, self.bias_mu).mean()
            sigma = F.softplus(F.linear(h, self.weight_sigma, self.bias_sigma)).mean()
            # mask
            epsilon = mu + sigma * torch.randn([c, n]).cuda()
            mask = torch.sigmoid(epsilon)

            if len(input.size()) == 3:  # [1,node,c]
                mask = mask.unsqueeze(dim=0)

            out = input.mul(mask).div(torch.sigmoid(mu.data / torch.sqrt(1. + 3.14 / 8. * sigma.data ** 2.)))
        else:
            out = input

        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs  to  be div by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  # the number of training examples
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            # Fills elements of self tensor with value where mask is True
        # print(energy.size)
        # print(energy)
        # print(self.embed_size)
        # print()
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TED(nn.Module):
    def __init__(self, input_size=3,
                 num_class=2,
                 hidden_size=64,
                 dropout=0.2,
                 max_length=22,
                 bias=False,
                 device=[]):
        super(TED, self).__init__()
        self.input_size = input_size
        self.device = device
        Layer = [hidden_size, hidden_size * 2, hidden_size * 4, hidden_size * 4, hidden_size * 2, hidden_size]
        self.feature_embedding = nn.Linear(input_size, Layer[0])
        self.spe_embedding = spe_feat_extract16(dropout=0.2)
        self.position_embedding = nn.Embedding(max_length, Layer[0] + 128)
        # self.adrop0 = AdvancedDropout(Layer[0]+128)

        self.T1 = TransformerBlock(Layer[0] + 128, heads=4, dropout=0.2, forward_expansion=1)
        self.G1 = GCN_RESBLOCK(Layer[0] + 128, Layer[1], bias=bias, norm='none')
        self.adrop1 = AdvancedDropout(Layer[1])

        self.T2 = TransformerBlock(Layer[1], heads=4, dropout=0.2, forward_expansion=1)
        self.G2 = GCN_RESBLOCK(Layer[1], Layer[2], bias=bias, norm='none')
        self.adrop2 = AdvancedDropout(Layer[2])

        self.T3 = TransformerBlock(Layer[2], heads=4, dropout=0.2, forward_expansion=1)
        # self.G3 = GCN_RESBLOCK(Layer[2], Layer[3], bias=bias)
        self.adrop3 = AdvancedDropout(Layer[3])

        self.layer_output = nn.Sequential(nn.Linear(Layer[2], num_class))

        self.dropout = nn.Dropout(dropout)

    def forward(self, swc, adj, spe):
        # print(adj.shape, swc.shape)
        if len(swc.shape) >= 3:
            swc = swc.squeeze(dim=0)
        N, C = swc.shape  # node_num, channel 3
        swc = swc.unsqueeze(dim=0)  # batch 1,
        # adj = adj.unsqueeze(dim=0)
        swc_feat = self.feature_embedding(swc)
        spe_feat = self.spe_embedding(spe)
        # print(swc_feat.shape,spe_feat.shape)
        # print(swc_feat)
        # print(spe_feat)
        feat_cat = torch.cat([swc_feat, spe_feat], dim=2)
        # print(feat_cat)
        pos_feat = self.position_embedding(torch.arange(0, N).expand(1, N).to(self.device))
        # print(pos_feat)
        input_feat = self.dropout(feat_cat + pos_feat)

        y1 = self.T1(input_feat, input_feat, input_feat)
        # print(adj.shape,y1.shape)
        y1 = self.adrop1(self.G1(adj, y1))

        # print('y1: ', y1.shape)
        y2 = self.T2(y1, y1, y1)
        y2 = self.adrop2(self.G2(adj, y2))

        y3 = self.T3(y2, y2, y2)
        y3 = self.adrop3(y3)  # self.G3(adj,y3)

        #
        # y4 = self.T4(y3,y3,y3)
        # y4 = self.G4(adj,y4)
        # print(y3.shape)
        out = torch.mean(y3, dim=1, keepdim=True)
        # print(out.shape)
        # print('out1: ', out)
        out = self.layer_output(out)
        # print("x: ", x[0])
        # print('out: ', out)
        return out.squeeze(dim=0)

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = RESGCN_SPE_TRANSFORM_ADROP(3, device=device).to(device)
#     feat_xyz = torch.randn((6, 3)).to(device)
#     adj = torch.randn((6, 6)).to(device)
#     feat_tif = torch.randn((6, 9, 16, 16)).to(device)
#     flops, params = profile(model, inputs=(adj,feat_xyz, feat_tif), verbose=False)
#     print('flops: ', flops, 'params: ', params)
#     print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
