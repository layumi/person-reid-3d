from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

# Swish from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py#L39
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SE(nn.Module):
    def __init__(self, input_dim):
        super(SE, self).__init__()
        num_squeezed_channels = max(1, int(input_dim/24.0))
        self.se_reduce = nn.Conv2d(in_channels=input_dim, out_channels=num_squeezed_channels, kernel_size=1)
        self.se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=input_dim, kernel_size=1)
        self.swish = MemoryEfficientSwish()

    def forward(self, x):
        x_squeezed = self.se_expand(self.swish(self.se_reduce(x)))
        x = torch.sigmoid(x_squeezed) * x
        return x

def build_shared_mlp(mlp_spec: List[int], nsample=1, norml='bn', activation = 'relu', use_se=True, use_neighbor=False, light=False):
    layers = []
    if norml == 'bn': 
        bias = False

    for i in range(1, len(mlp_spec)):
        if use_neighbor and i==1:
            kernel_size = (1, nsample)
        else:
            kernel_size = (1,1)

        if light and mlp_spec[i - 1] == mlp_spec[i]:
            layers.append(
                nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=kernel_size, groups=mlp_spec[i], bias=bias)
            )
        else: 
            layers.append(
                nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=kernel_size, bias=bias)
            )

        if norml=='bn':
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        elif norml=='in':
            layers.append(nn.InstanceNorm2d(mlp_spec[i]))
        

        if i == len(mlp_spec) - 1:
            if activation == 'swish':
                layers.append(MemoryEfficientSwish())
            elif activation == 'leaky':
                layers.append(nn.LeakyReLU(0.2, True))
            elif activation == 'relu':
                layers.append(nn.ReLU(True))

        if i == 1 and use_se:
            layers.append(SE(mlp_spec[i]))

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.fuse = 'concat'
    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
            )

        if self.fuse == 'add': 
            for i in range(len(self.groupers)):
                new_features = self.groupers[i](
                    xyz, new_xyz, features
                )  # (B, C, npoint, nsample)

                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                if i==0:
                    new_features_sum = new_features
                else:
                    new_features_sum += new_features
            return new_xyz, new_features_sum


        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    
    norml : string
        Use 'bn' or 'in'
    """

    def __init__(self, npoint, radii, nsamples, mlps, fuse= 'concat', norml='bn', activation = 'relu', use_se=True, use_xyz=True, use_neighbor=False, light=False):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.fuse = fuse
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, nsample, norml, activation, use_se, use_neighbor, light))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    norml : string
        Use 'bn' or 'in'
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, fuse= 'concat', norml='bn', activation = 'relu', use_se=True, use_xyz=True, use_neighbor=False, light=False
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            fuse = fuse,
            norml = norml,
            activation = activation,
            use_se = use_se,
            use_xyz=use_xyz,
            use_neighbor = use_neighbor,
            light = light
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    norml : string
        Use 'bn' or 'in'
    """

    def __init__(self, mlp, fuse= 'concat', norml='bn', activation = 'relu', use_se=True, use_neighbor=False, light=False):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, norml=norml, activation = activation, use_se=use_se, use_neighbor=use_neighbor, light=light)
        self.fuse = fuse

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)
