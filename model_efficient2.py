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
from market3d import Market3D
from utils import get_graph_feature, weights_init_kaiming, weights_init_classifier, drop_connect, farthest_point_sample, channel_shuffle
from ptflops import get_model_complexity_info
from functools import partial
from torch.utils.data import DataLoader


class EdgeConv_Light(EdgeConv):
    def __init__(self, in_feat, out_feat, batch_norm=False):
        super().__init__(in_feat, out_feat, batch_norm)
        self.theta = nn.Linear(in_feat, out_feat, bias = False)
        self.phi = nn.Linear(in_feat, out_feat, bias = False)

class ModelE(nn.Module):
    def __init__(self, k, feature_dims, emb_dims, output_classes, init_points = 512, input_dims=3,
                 dropout_prob=0.5, npart=1, id_skip=False, drop_connect_rate=0, res_scale = 1.0,
                 light = False, bias = False, cluster='xyz', conv='EdgeConv', use_xyz=True, 
                 use_se = True, graph_jitter = False, pre_act = False, norm = 'bn', stride=2, 
                 layer_drop = 0, num_conv=1, shuffle = 0 ):
        super(ModelE, self).__init__()

        self.npart = npart
        self.norm = norm
        self.shuffle = shuffle
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
        self.light = light
        self.pre_act = pre_act
        self.num_conv = num_conv
        #self.proj_in = nn.Linear(input_dims, input_dims)

        self.num_layers = len(feature_dims)
        npoint = init_points
        last_npoint = -1
        for i in range(self.num_layers):
            if k==1: 
                self.conv.append(nn.Conv2d(feature_dims[i-1] if i > 0 else input_dims, 
                                     feature_dims[i] , kernel_size=1, 
                                     bias = True))
                self.bn.append( nn.BatchNorm1d( feature_dims[i] ))
            elif conv == 'EdgeConv':
                for j in range(self.num_conv):
                    if j==0:
                        self.conv.append( nn.Conv2d(
                            feature_dims[i - 1]*2 if i > 0 else input_dims*2,
                            feature_dims[i], 
                            kernel_size=1,
                            groups = 2 if i>0 else 1,
                            bias = False ))
                    else: 
                        self.conv.append( nn.Conv2d(
                            feature_dims[i]*2, 
                            feature_dims[i],
                            kernel_size=1,
                            groups = 2, #feature_dims[i],
                            bias = False ))

                    if i==0 and j==0 and pre_act:
                        norm_dim = input_dims
                    else:
                        norm_dim = feature_dims[i-1] if pre_act and j==0 else feature_dims[i]

                    if norm == 'ln':
                        if layer_drop>0:
                            self.bn.append(nn.Sequential( 
                                 nn.LayerNorm(norm_dim),
                                 nn.Dropout(layer_drop)) ) 
                        else:
                            self.bn.append(
                                 nn.LayerNorm(norm_dim))
                    else:
                        if layer_drop>0:
                            self.bn.append(nn.Sequential( 
                                 nn.BatchNorm1d(norm_dim),
                                 nn.Dropout(layer_drop)) )
                        else:
                            self.bn.append(
                                 nn.BatchNorm1d(norm_dim))
                                

            if i>0 and feature_dims[i]>feature_dims[i-1]:
                npoint = npoint//stride

            if npoint != last_npoint:
                if id_skip:
                    self.conv_s2.append( nn.Conv1d(feature_dims[i-1] if i > 0 else input_dims, 
                                   feature_dims[i], kernel_size=1,
                                   groups = feature_dims[i-1] if i > 0 else input_dims,
                                   bias = False))
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
                    light = False
                ))
                last_npoint = npoint
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
                self.embs.append(nn.Linear(feature_dims[-1], 512, bias=bias))
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
        last_feature_dim = -1
        #h = self.proj_in(rgb)
        h = rgb
        s2_count = 0
        for i in range(self.num_layers):
            h_input = h.clone()
            xyz_input = xyz.clone()
            batch_size, n_points, feature_dim  = h.shape

            ######## Build Graph #########
            last_point = n_points

            ######### Dynamic Graph Conv #########
            xyz = xyz.transpose(1, 2).contiguous()
            #print(h.shape) # batchsize x point_number x feature_dim
            h = h.transpose(1, 2).contiguous()
            for j in range(self.num_conv):
                index = self.num_conv*i+j
                ####### BN + ReLU #####
                if self.pre_act == True:
                    if self.norm == 'ln':
                        h = h.transpose(1, 2).contiguous()
                        h = self.bn[index](h)
                        h = h.transpose(1, 2).contiguous()
                    else:
                        h = self.bn[index](h)
                    h = F.leaky_relu(h, 0.2)

                ####### Graph Feature ###########
                if self.k==1 and j==0:
                    h = h.unsqueeze(-1)
                else:
                    if i == self.num_layers-1:
                        if self.cluster == 'xyz':
                            h = get_graph_feature(xyz, h, k=self.k)
                        elif self.cluster == 'xyzrgb' or self.cluster == 'allxyzrgb':
                            h = get_graph_feature( torch.cat( (xyz, h), 1), h, k=self.k)
                    else:
                    # Common Layers
                        if self.cluster == 'allxyzrgb':
                            h = get_graph_feature( torch.cat( (xyz, h), 1), h, k=self.k)
                        else:
                            h = get_graph_feature(xyz, h, k=self.k)

                ####### Conv ##########
                h = self.conv[index](h)
                h = h.max(dim=-1, keepdim=False)[0]
                ####### BN + ReLU #####
                if self.pre_act == False:
                    if self.norm == 'ln':
                        h = h.transpose(1, 2).contiguous()
                        h = self.bn[index](h)
                        h = h.transpose(1, 2).contiguous()
                    else: 
                        h = self.bn[index](h)
                    h = F.leaky_relu(h, 0.2)


            ######### Residual Before Downsampling#############
            if self.id_skip==1: 
                if istrain and self.drop_connect_rate>0:
                    h = drop_connect(h, p=self.drop_connect_rate, training=istrain)
                if feature_dim != last_feature_dim:
                    h_input = self.conv_s2[s2_count](h_input)
                h = h_input + self.res_scale * h

            #print(h.shape) # batchsize x point_number x feature_dim
            batch_size, feature_dim, n_points  = h.shape

            ######### PointNet++ MSG ########
            if feature_dim != last_feature_dim:
                #h = h.transpose(1, 2).contiguous()
                xyz, h  = self.sa[s2_count](xyz_input, h)
                #h = h.transpose(1, 2).contiguous()
                if self.id_skip == 2: 
                    h_input = pointnet2_utils.gather_operation(
                        h_input.transpose(1, 2).contiguous(), 
                        pointnet2_utils.furthest_point_sample(xyz_input, h.shape[2] )
                    )
            else:
                xyz = xyz.transpose(1, 2).contiguous()
                h_input = h_input.transpose(1, 2).contiguous()

            ######### Residual After Downsampling (Paper) #############
            if self.id_skip==2:
                if istrain and self.drop_connect_rate>0:
                    h = drop_connect(h, p=self.drop_connect_rate, training=istrain)
                if feature_dim != last_feature_dim:
                    h_input = self.conv_s2[s2_count](h_input)
                h = h_input + self.res_scale * h

            if self.shuffle>0:
                h = channel_shuffle(h, self.shuffle)

            h = h.transpose(1, 2).contiguous()
            if feature_dim != last_feature_dim:
                s2_count +=1
                last_feature_dim = feature_dim

            #print(xyz.shape, h.shape)
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
            #batch_size, n_points, _ = h.shape
            #y_index = torch.argsort(xyz[:, :, 1],dim = 1).view(batch_size * n_points, -1)
            #h = h.view(batch_size * n_points, -1)
            #h = h[y_index, :].view(batch_size, n_points, -1)
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

class ModelE_dense2(ModelE):
    def __init__(self, k, feature_dims, emb_dims, output_classes, init_points = 512, input_dims=3,
                 dropout_prob=0.5, npart=1, id_skip=False, drop_connect_rate=0, res_scale=1.0, 
                 light=False, bias = False, cluster='xyz', conv='EdgeConv', use_xyz=True, 
                 use_se=True, graph_jitter = False, pre_act = False, norm = 'bn', stride=2,
                 layer_drop = 0, num_conv=1, shuffle=0, temp = False):
        super().__init__(k, feature_dims, emb_dims, output_classes, init_points, input_dims, 
                 dropout_prob, npart, id_skip, drop_connect_rate, res_scale, 
                 light, bias, cluster, conv, use_xyz, use_se, graph_jitter, pre_act, norm, stride,
                 layer_drop, num_conv, shuffle)
        self.sa = nn.ModuleList()
        npoint = init_points
        if temp: 
            self.logit_scale = nn.Parameter(torch.ones(()), requires_grad = True)
        last_npoint = -1
        for i in range(len(feature_dims)):
            if i>0 and feature_dims[i]>feature_dims[i-1]:
                npoint = npoint//stride

            rest_feature = feature_dims[i] - 2 * (feature_dims[i]//3)
            if npoint != last_npoint:
                self.sa.append( PointnetSAModuleMSG(
                    npoint=npoint,
                    radii = [0.1, 0.2, 0.4],
                    nsamples = [4, 8, 12],
                    mlps=[
                      [feature_dims[i], feature_dims[i]//3, feature_dims[i]//3],
                      [feature_dims[i], feature_dims[i]//3, feature_dims[i]//3],
                      [feature_dims[i], feature_dims[i]//3, rest_feature],
                    ],
                    fuse = 'concat', # fuse = 'add'
                    norml = 'bn',
                    activation = 'relu',
                    use_se = use_se,
                    use_xyz = use_xyz,
                    use_neighbor = False,
                    light = light
                )
                )
                last_npoint = npoint
        # since add 3 branch
        weights_init_kaiming2 = lambda x:weights_init_kaiming(x, L=self.num_layers)
        self.sa.apply(weights_init_kaiming2)

if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = ModelE_dense2( 5, [48, 96, 96, 192, 192, 384, 384], [512], stride=4, 
                        output_classes=751, cluster='xyzrgb', init_points = 512, 
                        input_dims=3, dropout_prob=0.5, npart= 1, id_skip=2,  
                        pre_act = False, norm = 'bn', layer_drop=0, num_conv=2, light=False,
                        shuffle = 3)
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
    #output = net(xyz, rgb)
    market_data = Market3D('./2DMarket', flip=True, slim=0.25, bg=True)

    CustomDataLoader = partial(
        DataLoader,
        num_workers=0,
        batch_size=8,
        shuffle=True,
        drop_last=True)
    query_loader = CustomDataLoader(market_data.query())
    batch0,label0 = next(iter(query_loader))
    batch0 = batch0[0].unsqueeze(0)
    print(batch0.shape)
    macs, params = get_model_complexity_info(net, batch0.cuda(), ((round(6890*0.5), 3)  ), as_strings=True, print_per_layer_stat=False, verbose=True)
#print(macs)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    #print(output.shape)
        
