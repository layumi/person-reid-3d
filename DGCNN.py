import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import KNNGraph, EdgeConv
import numpy as np
from utils import weights_init_kaiming, weights_init_classifier
# I modified the code from https://raw.githubusercontent.com/dmlc/dgl/master/examples/pytorch/pointcloud/model.py
class DGCNN(nn.Module):
    def __init__(self, k, feature_dims, emb_dims, output_classes, input_dims=3,
                 dropout_prob=0.5):
        super(DGCNN, self).__init__()

        self.nng = KNNGraph(k)
        self.conv = nn.ModuleList()

        self.num_layers = len(feature_dims)
        for i in range(self.num_layers):
            self.conv.append(EdgeConv(
                feature_dims[i - 1] if i > 0 else input_dims,
                feature_dims[i],
                batch_norm=True))

        self.proj = nn.Linear(sum(feature_dims), emb_dims[0])

        self.embs = nn.ModuleList()
        self.bn_embs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.num_embs = len(emb_dims) - 1
        for i in range(1, self.num_embs + 1):
            self.embs.append(nn.Linear(
                # * 2 because of concatenation of max- and mean-pooling
                emb_dims[i - 1] if i > 1 else (emb_dims[i - 1] * 2),
                emb_dims[i]))
            self.bn_embs.append(nn.BatchNorm1d(emb_dims[i]))
            self.dropouts.append(nn.Dropout(dropout_prob))

        self.proj_output = nn.Linear(emb_dims[-1], output_classes)
        # Init
        self.conv.apply(weights_init_kaiming)
        self.proj.apply(weights_init_kaiming)
        self.embs.apply(weights_init_kaiming)
        self.bn_embs.apply(weights_init_kaiming)
        self.proj_output.apply(weights_init_classifier)

    def forward(self, xyz, rgb, istrain=False):
        hs = []
        h = rgb
        batch_size, n_points, x_dims = h.shape
        g = self.nng(xyz)

        for i in range(self.num_layers):
            h = h.view(batch_size * n_points, -1)
            h = self.conv[i](g, h)
            h = F.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            hs.append(h)

        h = torch.cat(hs, 2)
        h = self.proj(h)
        h_max, _ = torch.max(h, 1)
        h_avg = torch.mean(h, 1)
        h = torch.cat([h_max, h_avg], 1)

        for i in range(self.num_embs):
            h = self.embs[i](h)
            h = self.bn_embs[i](h)
            h = F.leaky_relu(h, 0.2)
            h = self.dropouts[i](h)

        h = self.proj_output(h)
        return h

if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it.
#    net = Model_dense( 20, [64, 128, 256, 512], [512, 512], output_classes=751, init_points = 512, input_dims=3, dropout_prob=0.5, npart= 1)
    net = DGCNN( 20, [64,128,256,512], [512,512], output_classes=751,  input_dims=3, dropout_prob=0.5)
    xyz = torch.FloatTensor(np.random.normal(size=(4,6890, 3))).cuda()
    rgb = torch.FloatTensor(4, 6890, 3).cuda()
    net = net.cuda()
    print(net)
    net.proj_output = nn.Sequential()
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of parameters: %.2f M'% (params/1e6) )
    output = net(xyz, rgb)
    print('net output size:')
    print(output.shape)

