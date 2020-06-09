import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
from utils import weights_init_kaiming, weights_init_classifier
import numpy as np

class PointNet2SSG(nn.Module):
    def __init__(self, output_classes=751, init_points = 512, input_dims=3, dropout_prob=0.5, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint= init_points,
                radius=0.2,
                nsample=64,
                mlp=[input_dims, 64, 64, 128],
                use_xyz=use_xyz,
                use_se = False,
            )
        )
        #batchsize 512 128
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz = use_xyz,
                use_se = False
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], 
                use_xyz = use_xyz,
                use_se = False
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,True),
            nn.Linear(512, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_prob)
            )

        self.classifier = nn.Sequential(
            nn.Linear(512, output_classes)
        )
       # initial
        self.SA_modules.apply(weights_init_kaiming)
        self.fc_layer.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, xyz, rgb, istrain=False):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        features = rgb.transpose(1, 2).contiguous() 
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return self.classifier( self.fc_layer(features.squeeze(-1)) )

class PointNet2MSG(PointNet2SSG):
    def __init__(self, output_classes=751, init_points = 512, input_dims=3, dropout_prob=0.5, use_xyz=True):
        super().__init__( output_classes = output_classes, dropout_prob=dropout_prob)
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[3, 32, 32, 64], [3, 64, 64, 128], [3, 64, 96, 128]],
                use_xyz=use_xyz,
                use_se=False,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=use_xyz,
                use_se=False,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=use_xyz,
                use_se=False,
            )
        )
        self.SA_modules.apply(weights_init_kaiming)

if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it.
#    net = Model_dense( 20, [64, 128, 256, 512], [512, 512], output_classes=751, init_points = 512, input_dims=3, dropout_prob=0.5, npart= 1)
    net = PointNet2MSG(output_classes=751, init_points = 512, input_dims=3, dropout_prob=0.5 )
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

