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
from utils import get_graph_feature, weights_init_kaiming, weights_init_classifier, drop_connect, farthest_point_sample, channel_shuffle, L2norm
from ptflops import get_model_complexity_info
from functools import partial
from torch.utils.data import DataLoader

######################################################################
class GeM(nn.Module):
    # change to weighted sum
    def __init__(self, dim=1, p=0., eps=1e-6, cg = False, npart=1):
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones((npart))*p, requires_grad = True) #initial p
        self.npart = npart
        self.eps = eps
        self.dim = dim
        self.cg =cg
        if self.cg:
            self.gating  = ContextGating(dim)

    def forward(self, x):
        if self.cg:
            x = x.transpose(1,-1).contiguous()
            x = self.gating(x)
            x = x.transpose(1,-1).contiguous()
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        s = x.shape
        x_max = torch.nn.functional.adaptive_max_pool2d(x, (s[-2], self.npart)).view(-1, self.npart)
        x_avg = torch.nn.functional.adaptive_avg_pool2d(x, (s[-2], self.npart)).view(-1, self.npart)
        #x_max = x.max(dim=-1, keepdim=False)[0]
        #x_avg = x.mean(dim=-1, keepdim=False)
        w = torch.sigmoid(self.p)
        x = x_max*w + x_avg*(1-w)
        if self.npart==1:
            x = x.view(s[0:-1])
        elif len(s)==3:
            x = x.view( (s[0], s[1], self.npart))
        elif len(s)==4:
            x = x.view( (s[0], s[1], s[2], self.npart))
        return x

    def __repr__(self):
        if self.cg:
            return self.__class__.__name__ + '(' + 'p=' + '{:.2f}'.format(self.p[0]) + ', ' + 'cg=' + str(self.dim) + ')'
        if self.npart>1:
            s = ''
            for i in range(self.npart):
                s += self.__class__.__name__ + '(' + 'p=' + '{:.2f}'.format(self.p[i]) + '),'
            return s 
        return self.__class__.__name__ + '(' + 'p=' + '{:.2f}'.format(self.p[0]) + ')'


class ContextGating(nn.Module):
    def __init__(self, input_size):
        super(ContextGating, self).__init__()
        self.linear = nn.Sequential(*[nn.Linear(input_size, input_size//16, bias=True),
                                      nn.ReLU(),
                                      nn.Linear(input_size//16, input_size, bias=True)])
        self.linear.apply(weights_init_kaiming)

    def forward(self, input):
        s = input.shape
        x = input.view(-1, s[-1])
        wx = self.linear(x)
        # print(wx.size())
        gates = torch.sigmoid(wx)
        x = gates*x
        x = x.view(s)
        return x


class IBN1d(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN1d, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm1d(self.half, affine=True)
        self.BN = nn.BatchNorm1d(planes - self.half)
        self.IN.apply(weights_init_kaiming)
        self.BN.apply(weights_init_kaiming)
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class IBN2d(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN2d, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)
        self.IN.apply(weights_init_kaiming)
        self.BN.apply(weights_init_kaiming)
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class Conv_ASPP(nn.Module):
    def __init__(self, inplanes, outplanes, ASPP_type=1):
        super(Conv_ASPP, self).__init__()
        self.ASPP_type = ASPP_type
        if self.ASPP_type ==1:
            self.conv1 = nn.Sequential(*[nn.Conv2d( inplanes, outplanes - outplanes//2, kernel_size=1,bias = True ),
                                     nn.BatchNorm2d(outplanes - outplanes//2), GeM(outplanes - outplanes//2) ])
            self.conv2 = nn.Sequential(*[nn.Conv2d( inplanes, outplanes//2, kernel_size=1, bias = True ),
                                     nn.BatchNorm2d(outplanes//2), GeM(outplanes//2)])
        elif self.ASPP_type ==2:
            self.conv1 = nn.Sequential(*[nn.Conv2d( inplanes, outplanes - 2*outplanes//3, kernel_size=1,bias = True ),
                                     nn.BatchNorm2d(outplanes - 2*outplanes//3), GeM(outplanes - 2*outplanes//3) ])
            self.conv2 = nn.Sequential(*[nn.Conv2d( inplanes, outplanes//3, kernel_size=1, bias = True ),
                                     nn.BatchNorm2d(outplanes//3), GeM(outplanes//3)])
            self.conv3 = nn.Sequential(*[nn.Conv2d( inplanes, outplanes//3, kernel_size=1,bias = True ),
                                     nn.BatchNorm2d(outplanes//3), GeM(outplanes//3) ])
    def forward(self, x):
        B, C, N, neighbor = x.shape
        if self.ASPP_type ==1:
            x_half = x[:,:,:,0:neighbor//2].contiguous()
            out1 = self.conv1(x)
            out2 = self.conv2(x_half)
            out = torch.cat((out1, out2), 1)
        elif self.ASPP_type ==2:
            x_2 = x[:,:,:,0:neighbor//3].contiguous()
            x_3 = x[:,:,:,0:2*neighbor//3].contiguous()
            out1 = self.conv1(x)
            out2 = self.conv2(x_2)
            out3 = self.conv2(x_3)
            out = torch.cat((out1, out2, out3), 1)

        return out

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
                 layer_drop = 0, num_conv=1, temp = False, gem=False, cg=False, ASPP=0):
        super(ModelE, self).__init__()

        self.npart = npart
        self.norm = norm
        self.graph_jitter = graph_jitter
        self.res_scale = res_scale
        self.return_f = temp
        self.id_skip = id_skip
        self.drop_connect_rate = drop_connect_rate
        self.nng = KNNGraphE(k)  # with random neighbor
        self.conv = nn.ModuleList()
        self.ASPP = ASPP
        self.cg = cg
        if ASPP>0:
            norm = 'none'
        self.conv_s1 = nn.ModuleList()
        self.conv_s2 = nn.ModuleList()
        self.gem = gem
        if gem:
            self.agg = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.sa = nn.ModuleList()
        if id_skip:
            self.p_w = []
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
                group_num = 2 if light and i>0 else 1
                for j in range(self.num_conv):
                    if j==0:
                        self.conv.append( nn.Conv2d(
                            feature_dims[i - 1]*2 if i > 0 else input_dims*2,
                            feature_dims[i], 
                            kernel_size=1,
                            groups = group_num, 
                            bias = True ) if self.ASPP==0 else Conv_ASPP(
                            feature_dims[i - 1]*2 if i > 0 else input_dims*2,
                            feature_dims[i], self.ASPP) )
                    else: 
                        self.conv.append( nn.Conv2d(
                            feature_dims[i]*2, 
                            feature_dims[i],
                            kernel_size=1,
                            groups = group_num,
                            bias = True ) if self.ASPP==0 else Conv_ASPP(
                            feature_dims[i]*2,
                            feature_dims[i], self.ASPP))

                    if self.gem and self.ASPP==0:
                        self.agg.append(GeM(dim=feature_dims[i], cg = self.cg))

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
                    elif norm == 'ibn':
                        if layer_drop>0:
                            self.bn.append(nn.Sequential(
                                 IBN1d(norm_dim),
                                 nn.Dropout(layer_drop)) )
                        else:
                            self.bn.append(
                                 IBN1d(norm_dim))
                    elif norm == 'ibn2':
                        if layer_drop>0:
                            self.bn.append(nn.Sequential(
                                 IBN2d(norm_dim),
                                 nn.Dropout(layer_drop)) )
                        else:
                            self.bn.append(
                                 IBN2d(norm_dim))
                    elif norm == 'bn':
                        if layer_drop>0:
                            self.bn.append(nn.Sequential( 
                                 nn.BatchNorm1d(norm_dim),
                                 nn.Dropout(layer_drop)) )
                        else:
                            self.bn.append(
                                 nn.BatchNorm1d(norm_dim))
                    elif norm == 'bn2':
                        if layer_drop>0:
                            self.bn.append(nn.Sequential(
                                 nn.BatchNorm2d(norm_dim),
                                 nn.Dropout(layer_drop)) )
                        else:
                            self.bn.append(
                                 nn.BatchNorm2d(norm_dim))
                    elif norm == 'none':
                        self.bn.append(nn.Sequential())
                    else: 
                        print('!!! UNknown Normalization Layer')

            if i>0 and feature_dims[i]>feature_dims[i-1]:
                npoint = npoint//stride

            if id_skip:
                self.p_w = nn.Parameter(torch.ones((self.num_layers))*0, requires_grad = True) 

            if npoint != last_npoint:
                if id_skip:
                    self.conv_s2.append( nn.Sequential(*[ 
                                   nn.Linear(feature_dims[i-1] if i > 0 else input_dims, 
                                   feature_dims[i]), nn.LeakyReLU(0.2)])) 
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

        if self.cg:
            self.gating = ContextGating(feature_dims[-1])
            self.gating.apply(weights_init_kaiming)

        if self.npart == 1: 
            self.embs.append(nn.Linear(
                # * 2 because of concatenation of max- and mean-pooling
                feature_dims[-1]*2, emb_dims[0], bias=bias))
            self.bn_embs.append(nn.BatchNorm1d(emb_dims[0]))
            self.dropouts.append(nn.Dropout(dropout_prob, inplace=True))
            self.proj_output = nn.Linear(emb_dims[0], output_classes)
            self.proj_output.apply(weights_init_classifier)
        else: 
            self.globpool = GeM()
            self.partpool = GeM(npart = self.npart)
            self.proj_outputs = nn.ModuleList()
            for i in range(0, self.npart+1):  # one more for global
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
                if self.light == True and i >0:
                    #shuffle after the first layer
                    h = channel_shuffle(h, 2)
                    h = self.conv[index](h)
                else:
                    h = self.conv[index](h)
                ##### BN2d before Aggregation ###
                if self.pre_act == False:
                    if self.norm =='bn2' or  self.norm =='ibn2':
                        h = self.bn[index](h)

                ###### Aggregation ####
                if self.ASPP==0:
                    if self.gem:
                        h = self.agg[index](h)
                    else:
                        h = h.max(dim=-1, keepdim=False)[0]

                ####### BN + ReLU #####
                if self.pre_act == False:
                    if self.norm == 'ln':
                        h = h.transpose(1, 2).contiguous()
                        h = self.bn[index](h)
                        h = h.transpose(1, 2).contiguous()
                    elif self.norm=="bn" or self.norm=="ibn": 
                        h = self.bn[index](h)
                    h = F.leaky_relu(h, 0.2)

            h = h.transpose(1, 2).contiguous()
            #print(h.shape) # batchsize x point_number x feature_dim
            batch_size, n_points, feature_dim  = h.shape

            if self.id_skip:
                p_w = self.p_w[i]
                res_w = torch.sigmoid(p_w) 

            ######### Residual Before Downsampling#############
            if self.id_skip==1: 
                if istrain and self.drop_connect_rate>0:
                    h = drop_connect(h, p=self.drop_connect_rate, training=istrain)
                if feature_dim != last_feature_dim:
                    h_input = self.conv_s2[s2_count](h_input)
                h = res_w*h_input + (1-res_w)*self.res_scale * h

            ######### PointNet++ MSG ########
            if feature_dim != last_feature_dim:
                h = h.transpose(1, 2).contiguous()
                xyz, h  = self.sa[s2_count](xyz_input, h)
                h = h.transpose(1, 2).contiguous()
                if self.id_skip == 2: 
                    h_input = pointnet2_utils.gather_operation(
                        h_input.transpose(1, 2).contiguous(), 
                        pointnet2_utils.furthest_point_sample(xyz_input, h.shape[1] )
                    ).transpose(1, 2).contiguous()
            else:
                xyz = xyz.transpose(1, 2).contiguous()

            ######### Residual After Downsampling (Paper) #############
            if self.id_skip==2:
                if istrain and self.drop_connect_rate>0:
                    h = drop_connect(h, p=self.drop_connect_rate, training=istrain)
                if feature_dim != last_feature_dim:
                    h_input = self.conv_s2[s2_count](h_input)
                h = res_w*h_input + (1-res_w)*self.res_scale * h

            if feature_dim != last_feature_dim:
                s2_count +=1
                last_feature_dim = feature_dim

            #print(xyz.shape, h.shape)
        if self.cg:
            h = self.gating(h)

        if self.npart==1:
            # Pooling
            h_max, _ = torch.max(h, 1)
            h_avg = torch.mean(h, 1)
            #hs.append(h_max)
            #hs.append(h_avg)

            h = torch.cat([h_max, h_avg], 1)
            h = self.embs[0](h)
            hf = self.bn_embs[0](h)
            h = self.dropouts[0](hf)
            h = self.proj_output(h)
        else:
            # original loss
            h0 = self.globpool(h.transpose(1, 2))
            h0 = self.embs[-1](h0)
            hf0 = self.bn_embs[-1](h0)
            h0 = self.dropouts[-1](hf0)
            h0 = self.proj_outputs[-1](h0)
            # Sort 
            batch_size, n_points, _ = h.shape
            y_bias = torch.argsort(xyz[:, :, 1], dim = 1) .view(batch_size * n_points)
            h = h.view(batch_size * n_points, -1)
            y_index = y_bias + torch.arange(0, n_points*batch_size,device='cuda')//n_points * n_points
            h = h[y_index, :].view(batch_size, n_points, -1)
            h = h.transpose(1, 2)
            # Part Pooling            
            h = self.partpool(h)
            hf = [L2norm(hf0)]
            hs = [h0]
            for i in range(0, self.npart):
                part_h = h[:,:,i]
                part_h = self.embs[i](part_h)
                part_hf = self.bn_embs[i](part_h)
                part_h = self.dropouts[i](part_hf)
                part_h = self.proj_outputs[i](part_h)
                hs.append(part_h)
                hf.append(L2norm(part_hf))
            h = hs
            #hf = torch.cat(hf, 1)
        if self.return_f:
            return [h, hf] 
        return h

class ModelE_dense(ModelE):
    def __init__(self, k, feature_dims, emb_dims, output_classes, init_points = 512, input_dims=3,
                 dropout_prob=0.5, npart=1, id_skip=False, drop_connect_rate=0, res_scale=1.0, 
                 light=False, bias = False, cluster='xyz', conv='EdgeConv', use_xyz=True, 
                 use_se=True, graph_jitter = False, pre_act = False, norm = 'bn', stride=2,
                 layer_drop = 0, num_conv=1, temp = False, gem=False, cg=False, ASPP=0):
        super().__init__(k, feature_dims, emb_dims, output_classes, init_points, input_dims, 
                 dropout_prob, npart, id_skip, drop_connect_rate, res_scale, 
                 light, bias, cluster, conv, use_xyz, use_se, graph_jitter, pre_act, norm, stride,
		 layer_drop, num_conv, temp, gem, cg, ASPP)
        self.sa = nn.ModuleList()
        npoint = init_points
        if temp:
            self.logit_scale = nn.Parameter(torch.ones(()), requires_grad = True)
        last_npoint = -1
        for i in range(len(feature_dims)):
            if i>0 and feature_dims[i]>feature_dims[i-1]:
                npoint = npoint//stride

            if npoint != last_npoint:
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
                    light = False
                )
                )
                last_npoint = npoint
        # since add 3 branch
        weights_init_kaiming2 = lambda x:weights_init_kaiming(x, L=self.num_layers)
        self.sa.apply(weights_init_kaiming2)

if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    #torch.backends.cudnn.enabled = False
    net = ModelE_dense( 20, [48, 96, 192,384], [512],  
                        output_classes=751, cluster='xyzrgb', init_points = 768, 
                        input_dims=3, dropout_prob=0.5, npart= 4, id_skip=0,  
                        pre_act = False, norm = 'bn2', layer_drop=0.1, num_conv=1, light=False,
                        temp=False, gem=True, cg=True, ASPP=0)
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
    market_data = Market3D('./2DMarket', flip=True, slim=0.5, bg=True)

    CustomDataLoader = partial(
        DataLoader,
        num_workers=0,
        batch_size=4,
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
        
