import torch
from torch.nn import init
import torch.nn as nn

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, epsilon=0.05, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        num_classes = targets.shape[-1]
        targets = (1 - self.epsilon) * targets + self.epsilon 
        loss = (- targets * log_probs).sum(1).mean()
        return loss

def weights_init_kaiming(m, L=1):
    classname = m.__class__.__name__
    # https://arxiv.org/pdf/1901.09321.pdf
    factor = L**(-0.5)
    if classname.find('Conv2') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') * factor  # For old pytorch, you may use kaiming_normal.
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=1e-6)
        init.constant_(m.bias.data, 0.0)

def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

def farthest_point_sample(x, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, C]
    """
    B, N, C = x.shape
    S = npoint
    y = torch.zeros(B, S, C).cuda()
    distance = torch.ones(B, N).cuda() * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).cuda()
    batch_indices = torch.arange(B, dtype=torch.long).cuda()
    for i in range(S):
        centroid = x[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((x - centroid)**2, -1)
        distance[dist < distance] = dist[dist < distance]
        farthest = torch.max(distance, -1)[1]
        y[:,i,:] = centroid.view(B, C)
    return y


