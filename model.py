import torch
import numpy

from torch import linalg


#region classical principal component analysis (PCA)
class PCAobj:
    dim: int
    k: int

    def __init__(self, dim=None, k=None):
        if k is None:
            k = dim

        assert k <= dim
        assert k > 0
        self.dim = dim
        self.k = k

        # PCA output
        self.s = None # singular values
        self.U = None # left singular matrix
        self.V = None # right singular matrix

    def fit(self, K):
        _, s, Vh = linalg.svd(K)
        self.s = s # eigenvalues
        self.V = torch.t(Vh)  # eigenvectors

#endregion