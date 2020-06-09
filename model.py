import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from dgl.nn.pytorch import KNNGraph, EdgeConv, GATConv, GraphConv, SAGEConv, SGConv, GatedGraphConv
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
from pointnet2_ops import pointnet2_utils 
from gated_gcn_layer import GatedGCNLayer
from KNNGraphE import KNNGraphE
import numpy as np
from utils import weights_init_kaiming, weights_init_classifier, drop_connect, farthest_point_sample

class EdgeConv_Light(EdgeConv):
    def __init__(self, in_feat, out_feat, batch_norm=False):
        super().__init__(in_feat, out_feat, batch_norm)
        self.theta = nn.Linear(in_feat, out_feat, bias = False)
        self.phi = nn.Linear(in_feat, out_feat, bias = False)

class Model(nn.Module):
    def __init__(self, k, feature_dims, emb_dims, output_classes, init_points = 512, input_dims=3,
                 dropout_prob=0.5, npart=1, id_skip=False, drop_connect_rate=0, res_scale = 1.0,
                 light = False, bias = False, cluster='xyz', conv='EdgeConv', use_xyz=True, use_se = True, graph_jitter = False):
        super(Model, self).__init__()

        self.npart = npart
        self.graph_jitter = graph_jitter
        self.res_scale = res_scale
        self.id_skip = id_skip
        self.drop_connect_rate = drop_connect_rate
        self.nng = KNNGraphE(k)  # with random neighbor
        self.conv = nn.ModuleList()
        self.conv_s1 = nn.ModuleList()
        self.conv_s2 = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.sa = nn.ModuleList()
        self.cluster = cluster
        self.feature_dims = feature_dims
        self.conv_type = conv
        self.init_points = init_points
        self.k = k
        #self.proj_in = nn.Linear(input_dims, input_dims)

        self.num_layers = len(feature_dims)
        npoint = init_points
        for i in range(self.num_layers):
            if k==1: 
                    self.conv.append(nn.Linear(feature_dims[i-1] if i > 0 else input_dims, 
                                     feature_dims[i] ))
            elif conv == 'EdgeConv':
                if light:
                    self.conv.append(EdgeConv_Light(
                        feature_dims[i - 1] if i > 0 else input_dims,
                        feature_dims[i],
                        batch_norm=True))
                else: 
                    self.conv.append(EdgeConv(
                        feature_dims[i - 1] if i > 0 else input_dims,
                        feature_dims[i],
                        batch_norm=True))
            elif conv == 'GATConv':
                    self.conv.append(GATConv(
                        feature_dims[i - 1] if i > 0 else input_dims,
                        feature_dims[i],
                        feat_drop=0.2, attn_drop=0.2,
                        residual=True,
                        num_heads=1))
            elif conv == 'GraphConv':
                    self.conv.append( GraphConv(
                        feature_dims[i - 1] if i > 0 else input_dims,
                        feature_dims[i]))
            elif conv == 'SAGEConv':
                    self.conv.append( SAGEConv(
                        feature_dims[i - 1] if i > 0 else input_dims,
                        feature_dims[i],
                        feat_drop=0.2,
                        aggregator_type='mean', 
                        norm = nn.BatchNorm1d(feature_dims[i])
                        ) )
            elif conv == 'SGConv':
                    self.conv.append( SGConv(
                        feature_dims[i - 1] if i > 0 else input_dims,
                        feature_dims[i]) )
            elif conv == 'GatedGCN': # missing etypes
                    self.conv.append( GatedGCNLayer(
                        feature_dims[i - 1] if i > 0 else input_dims,
                        feature_dims[i], 
                        dropout=0.0, 
                        graph_norm=True, batch_norm=True, residual=True)
                        )


            if i>0 and feature_dims[i]>feature_dims[i-1]:
                npoint = npoint//2
                if id_skip and  npoint <= self.init_points//4: # Only work on high level
                    self.conv_s2.append( nn.Linear(feature_dims[i-1], feature_dims[i] ))

            self.sa.append(PointnetSAModule(
                npoint=npoint,
                radius=0.2,
                nsample=64,
                mlp=[feature_dims[i], feature_dims[i], feature_dims[i]],
                fuse = 'add',
                norml = 'bn',
                activation = 'relu',
                use_se = use_se,
                use_xyz = use_xyz,
                use_neighbor = False,
                light = light
            ))
            #if id_skip:
            #    self.conv_s1.append( nn.Linear(feature_dims[i], feature_dims[i] ))

        self.embs = nn.ModuleList()
        self.bn_embs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.partpool =  nn.AdaptiveAvgPool1d(self.npart)
        if self.npart == 1: 
            self.embs.append(nn.Linear(
                # * 2 because of concatenation of max- and mean-pooling
                feature_dims[-1]*2, emb_dims[0], bias=bias))
            self.bn_embs.append(nn.BatchNorm1d(emb_dims[0]))
            self.dropouts.append(nn.Dropout(dropout_prob, inplace=True))
            self.proj_output = nn.Linear(emb_dims[0], output_classes)
            self.proj_output.apply(weights_init_classifier)
        else: 
            self.proj_outputs = nn.ModuleList()
            for i in range(0, self.npart):
                self.embs.append(nn.Linear(512, 512, bias=bias))
                self.bn_embs.append(nn.BatchNorm1d(512))
                self.dropouts.append(nn.Dropout(dropout_prob, inplace=True))
                self.proj_outputs.append(nn.Linear(512, output_classes))
            self.proj_outputs.apply(weights_init_classifier)

        # initial
        #self.proj_in.apply(weights_init_kaiming)
        self.conv.apply(weights_init_kaiming)
        self.conv_s1.apply(weights_init_kaiming)
        self.conv_s2.apply(weights_init_kaiming)
        weights_init_kaiming2 = lambda x:weights_init_kaiming(x,L=self.num_layers)
        self.sa.apply(weights_init_kaiming2) 
        #self.proj.apply(weights_init_kaiming)
        self.embs.apply(weights_init_kaiming)
        self.bn.apply(weights_init_kaiming)
        self.bn_embs.apply(weights_init_kaiming)
        self.npart = npart

    def forward(self, xyz, rgb, istrain=False):
        hs = []
        #xyz_copy = xyz.clone()
        #rgb_copy = rgb.clone()
        batch_size, n_points, _ = xyz.shape
        part_length = n_points//self.npart
        last_point = -1
        #h = self.proj_in(rgb)
        h = rgb
        s2_count = 0
        for i in range(self.num_layers):
            h_input = h.clone()
            xyz_input = xyz.clone()
            batch_size, n_points, _ = h.shape
            if self.k>1:
                if i == self.num_layers-1:
                    if self.cluster == 'xyz':
                        g = self.nng(xyz, istrain = istrain and self.graph_jitter)
                    elif self.cluster == 'rgb':
                        g = self.nng(h, istrain=istrain and self.graph_jitter)
                    elif self.cluster == 'xyzrgb':
                        g = self.nng( torch.cat((xyz,h), 2), istrain=istrain and self.graph_jitter)
                elif i==0 or  n_points !=  last_point:
                    g = self.nng(xyz, istrain=istrain and self.graph_jitter)
            last_point = n_points
            h = h.view(batch_size * n_points, -1)

            if self.k==1:
                h = self.conv[i](h)
            elif self.conv_type == 'GatedGCN':
                h = self.conv[i](g, h, g.edata['feat'], snorm_n = 1/g.number_of_nodes() , snorm_e = 1/g.number_of_edges())
            else:
                h = self.conv[i](g, h)
            h = F.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            h = h.transpose(1, 2).contiguous()
            xyz, h  = self.sa[i](xyz_input, h)
            h = h.transpose(1, 2).contiguous()
            #h = self.conv_s1[i](h)
            if self.id_skip and  h.shape[1] <= self.init_points//4:
            # We could use identity mapping Here or add connect drop
                if istrain and self.drop_connect_rate>0:
                    h = drop_connect(h, p=self.drop_connect_rate, training=istrain)

                if h.shape[1] == n_points:
                    h = h_input + self.res_scale * h  # Here I borrow the idea from Inception-ResNet-v2
                elif h.shape[1] == n_points//2:
                    h_input_s2 = pointnet2_utils.gather_operation(
                        h_input.transpose(1, 2).contiguous(), 
                        pointnet2_utils.furthest_point_sample(xyz_input, h.shape[1] )
                    ).transpose(1, 2).contiguous()
                    h = self.conv_s2[s2_count](h_input_s2) + self.res_scale * h
                    s2_count +=1
        if self.npart==1:
            # Pooling
            h_max, _ = torch.max(h, 1)
            h_avg = torch.mean(h, 1)
            hs.append(h_max)
            hs.append(h_avg)

            h = torch.cat(hs, 1)
            h = self.embs[0](h)
            h = self.bn_embs[0](h)
            h = self.dropouts[0](h)
            h = self.proj_output(h)
        else:
            # Sort 
            batch_size, n_points, _ = h.shape
            y_index = torch.argsort(xyz[:, :, 1],dim = 1).view(batch_size * n_points, -1)
            h = h.view(batch_size * n_points, -1)
            h = h[y_index, :].view(batch_size, n_points, -1)
            h = h.transpose(1, 2)
            # Part Pooling            
            h = self.partpool(h)
            for i in range(self.npart):
                part_h = h[:,:,i]
                part_h = self.embs[i](part_h)
                part_h = self.bn_embs[i](part_h)
                part_h = self.dropouts[i](part_h)
                part_h = self.proj_outputs[i](part_h)
                hs.append(part_h)
            h = hs
        return h

class Model_dense(Model):
    def __init__(self, k, feature_dims, emb_dims, output_classes, init_points = 512, input_dims=3,
                 dropout_prob=0.5, npart=1, id_skip=False, drop_connect_rate=0, res_scale=1.0, 
                 light=False, bias = False, cluster='xyz', conv='EdgeConv', use_xyz=True, 
                 use_se=True, graph_jitter = False):
        super().__init__(k, feature_dims, emb_dims, output_classes, init_points, input_dims, 
                 dropout_prob, npart, id_skip, drop_connect_rate, res_scale, 
                 light, bias, cluster, conv, use_xyz, use_se, graph_jitter)
        self.sa = nn.ModuleList()
        npoint = init_points
        for i in range(self.num_layers):
                if i>0 and feature_dims[i]>feature_dims[i-1]:
                    npoint = npoint//2
                self.sa.append( PointnetSAModuleMSG(
                    npoint=npoint,
                    radii = [0.1, 0.2, 0.4],
                    nsamples = [8, 16, 32],
                    mlps=[
                      [feature_dims[i], feature_dims[i]//2, feature_dims[i]],
                      [feature_dims[i], feature_dims[i]//2, feature_dims[i]],
                      [feature_dims[i], feature_dims[i]//2, feature_dims[i]],
                    ],
                    fuse = 'add', # fuse = 'add'
                    norml = 'bn',
                    activation = 'relu',
                    use_se = use_se,
                    use_xyz = use_xyz,
                    use_neighbor = False,
                    light = light
                )
                )
        # since add 3 branch
        weights_init_kaiming2 = lambda x:weights_init_kaiming(x, L=self.num_layers)
        self.sa.apply(weights_init_kaiming2)

class Model_dense2(Model):
    def __init__(self, k, feature_dims, emb_dims, output_classes, init_points = 512, input_dims=3,
                 dropout_prob=0.5, npart=1, id_skip=False, drop_connect_rate=0, res_scale=1.0,
                 light=False, bias = False, cluster='xyz', conv='EdgeConv', use_xyz=True, graph_jitter = False):
        super().__init__(k, feature_dims, emb_dims, output_classes, init_points, input_dims,
                 dropout_prob, npart, id_skip, drop_connect_rate, res_scale,
                 light, bias, cluster, conv, use_xyz, graph_jitter)
        self.sa = nn.ModuleList()
        npoint = init_points
        for i in range(self.num_layers):
            if i>0 and feature_dims[i]>feature_dims[i-1]:
                npoint = npoint//2
            self.sa.append( PointnetSAModuleMSG(
                npoint=npoint,
                radii = [0.1, 0.2, 0.4],
                nsamples = [8, 16, 32],
                mlps=[
                  [feature_dims[i], feature_dims[i]//2, feature_dims[i]//4],
                  [feature_dims[i], feature_dims[i]//2, feature_dims[i]//4],
                  [feature_dims[i], feature_dims[i]//2, feature_dims[i]//2],
                ],
                fuse = 'concat', # fuse = 'add'
                norml = 'bn',
                activation = 'relu',
                use_se = True,
                use_xyz = use_xyz,
                use_neighbor = False,
                light = light
            )
            )
        # since add 3 branch
        weights_init_kaiming2 = lambda x:weights_init_kaiming(x, L=self.num_layers)
        self.sa.apply(weights_init_kaiming2)


if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = Model_dense( 20, [48, 96, 96, 192, 192, 192, 192, 192, 384, 384], [512], output_classes=751, init_points = 512, input_dims=3, dropout_prob=0.5, npart= 1, id_skip=True)
#    net = Model_dense( 20, [40,40,80,80,192,192,320,320, 512], [512], output_classes=751, 
#                     init_points = 512, input_dims=3, dropout_prob=0.5, npart= 1, id_skip=True, 
#                     light=True, cluster='xyz', conv='SAGEConv', use_xyz=False)
    xyz = torch.FloatTensor(np.random.normal(size=(4, 6890, 3))).cuda()
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
        
