import torch.nn as nn
from dgl.graph import DGLGraph
from dgl import backend as F
from scipy import sparse
import numpy as np
def pairwise_squared_distance(x):
    '''
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    '''
    x2s = (x * x).sum(-1, keepdim=True)
    return x2s + x2s.transpose(-1, -2) - 2 * x @ x.transpose(-1, -2)


def knn_graphE(x, k, istrain=False):
    """Transforms the given point set to a directed graph, whose coordinates
    are given as a matrix. The predecessors of each point are its k-nearest
    neighbors.

    If a 3D tensor is given instead, then each row would be transformed into
    a separate graph.  The graphs will be unioned.

    Parameters
    ----------
    x : Tensor
        The input tensor.

        If 2D, each row of ``x`` corresponds to a node.

        If 3D, a k-NN graph would be constructed for each row.  Then
        the graphs are unioned.
    k : int
        The number of neighbors

    Returns
    -------
    DGLGraph
        The graph.  The node IDs are in the same order as ``x``.
    """
    if F.ndim(x) == 2:
        x = F.unsqueeze(x, 0)
    n_samples, n_points, _ = F.shape(x)

    dist = pairwise_squared_distance(x)
    if istrain and np.random.rand()>0.5:
        k_indices = F.argtopk(dist, round(1.5*k), 2, descending=False)
        rand_k = np.random.permutation( round(1.5*k)-1) [0:k-1] +1 # 0 + random k-1 
        rand_k = np.append(rand_k,0)
        k_indices = k_indices[:,:, rand_k] # add 0
    else:
        k_indices = F.argtopk(dist, k, 2, descending=False)

    dst = F.copy_to(k_indices, F.cpu())

    src = F.zeros_like(dst) + F.reshape(F.arange(0, n_points), (1, -1, 1))

    per_sample_offset = F.reshape(F.arange(0, n_samples) * n_points, (-1, 1, 1))
    dst += per_sample_offset
    src += per_sample_offset
    dst = F.reshape(dst, (-1,))
    src = F.reshape(src, (-1,))
    adj = sparse.csr_matrix((F.asnumpy(F.zeros_like(dst) + 1), (F.asnumpy(dst), F.asnumpy(src))))

    g = DGLGraph(adj, readonly=True)
    return g


class KNNGraphE(nn.Module):
    r"""Layer that transforms one point set into a graph, or a batch of
    point sets with the same number of points into a union of those graphs.

    If a batch of point set is provided, then the point :math:`j` in point
    set :math:`i` is mapped to graph node ID :math:`i \times M + j`, where
    :math:`M` is the number of nodes in each point set.

    The predecessors of each node are the k-nearest neighbors of the
    corresponding point.

    Parameters
    ----------
    k : int
        The number of neighbors
    """
    def __init__(self, k):
        super(KNNGraphE, self).__init__()
        self.k = k
    def forward(self, x, istrain=False):
        """Forward computation.

        Parameters
        ----------
        x : Tensor
            :math:`(M, D)` or :math:`(N, M, D)` where :math:`N` means the
            number of point sets, :math:`M` means the number of points in
            each point set, and :math:`D` means the size of features.

        Returns
        -------
        DGLGraph
            A DGLGraph with no features.
        """
        return knn_graphE(x, self.k, istrain)

