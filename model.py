import torch
import numpy

from torch import linalg


#region [PCA] classical principal component analysis
class PCAobj:
    def __init__(self, dim:int, k:int=None):
        if k is None:
            k = dim

        assert k <= dim
        assert k > 0
        self.dim = dim
        self.k = k

        # PCA output
        self.s = None # singular values
        self.V = None # right singular matrix

        # select largest k eigenpairs
        self.s_topk = None
        self.V_topk = None

    def fit(self, K:torch.Tensor):
        _, s, Vh = linalg.svd(K)
        self.s = s # eigenvalues
        self.V = torch.t(Vh)  # eigenvectors
        self.s_topk = self.s[0:self.k]
        self.V_topk = self.V[:, 0:self.k]

    def get_pseudodata(self, K:torch.Tensor=None):
        # K is optional
        if K is not None:
            self.fit(K)
        # Find X such that K = t(X)*X
        X = torch.mm(self.V, torch.diag(torch.sqrt(self.s)))
        X = torch.t(X)
        return X
#endregion
