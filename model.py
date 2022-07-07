import torch
import numpy
import warnings

from torch import linalg


# region [PCA] classical principal component analysis
class PCAobj:
    def __init__(self, dim: int, k: int = None):
        if k is None:
            k = dim

        assert k <= dim
        assert k > 0
        self.dim = dim
        self.k = k

        # PCA output
        self.s = None  # singular values
        self.V = None  # right singular matrix

        # select largest k eigenpairs
        self.s_topk = None
        self.V_topk = None

    def fit(self, K: torch.Tensor):
        _, s, Vh = linalg.svd(K)
        self.s = s  # eigenvalues
        self.V = torch.t(Vh)  # eigenvectors
        self.s_topk = self.s[0:self.k]
        self.V_topk = self.V[:, 0:self.k]

    def get_pseudodata(self, K: torch.Tensor = None):
        # K is optional
        if K is not None:
            self.fit(K)
        # Find X such that K = t(X)*X
        X = torch.mm(self.V, torch.diag(torch.sqrt(self.s)))
        X = torch.t(X)
        return X


# endregion


# region [2.3. Adjusted variance explained by PCs] H. Shen, J.Z. Huang (2008).
class VarExplain:
    def __init__(self, k: int = 10):
        self.k = k
        self.adj_var = torch.zeros(k)
        self.cvar = torch.zeros(k)
        self.cpev = None

    def fit(self, data: torch.Tensor, Z: torch.Tensor):
        for i in range(self.k):
            # compute adjusted variance
            data_proj = torch.mm(data, torch.mm(Z, torch.pinverse(Z)))
            _, R = linalg.qr(data_proj)
            self.cvar[i] = torch.sum(torch.square(torch.diag(R)))
            if i > 0:
                self.adj_var[i] = self.cvar[i] - self.cvar[i - 1]

        # compute cumulative percentage of explained variance (CPEV)
        self.adj_var[0] = self.cvar[0]
        _, R = linalg.qr(data)
        self.cpev = self.cvar[self.k - 1] / torch.sum(torch.square(torch.diag(R)))


# endregion


# region
class SparseDeflate:
    def __init__(self, dim: int = 0, method: str = 'hotelling', is_data: bool = False,
                 is_ortho: bool = False):
        # method specification
        self.method = method
        self.is_data = is_data
        self.is_ortho = is_ortho

        # initialize orthogonal basis and counter
        self.count = None
        self.Q = None
        if self.is_ortho:
            self.count = 0
            self.Q = torch.zeros(dim, dim)

    def fit(self, K: torch.Tensor, x: torch.Tensor, data: torch.Tensor = None):
        # SPCA deflation
        if self.method == 'hotelling':
            assert self.is_data is False, f'Hotelling method does not support data matrix.'
            if self.is_ortho:
                K, self.Q, self.count = self._ortho_hotelling(K, x, self.count, self.Q)
            else:
                K = self._hotelling(K, x)
            return K

        elif self.method == 'projection':
            if self.is_data:
                data = self._projection_data(data, x)
                return data
            else:
                if self.is_ortho:
                    K, self.Q, self.count = self._ortho_projection(K, x, self.count, self.Q)
                else:
                    K = self._projection(K, x)
                return K

        elif self.method == 'schur':
            if self.is_data:
                data = self._schur_data(data, x)
                return data
            else:
                K = self._schur(K, x)
                return K

        # Raise warning
        raise warnings.warn('Invalid parameters, no deflation is made.')

    # Hotelling's deflation
    @staticmethod
    def _hotelling(K: torch.Tensor, x: torch.Tensor):
        xKx = torch.inner(x, torch.matmul(K, x))
        K -= torch.outer(x, x) * xKx
        return K

    # Orthogonalized Hotelling's deflation
    @staticmethod
    def _ortho_hotelling(K: torch.Tensor, x: torch.Tensor, count: int,
                         Q: torch.Tensor):
        # standard Hotelling's deflation for the first round
        if count == 0:
            Q[:, count] = x
            return SparseDeflate()._hotelling(K, x)

        # OHD for the subsequent round
        else:
            # Gram-Schmidt process
            Q[:, count] = x - torch.matmul(
                torch.mm(Q[:, 0:count], torch.t(Q[:, 0:count])),
                x)
            Q[:, count] /= torch.norm(Q[:, count])
            return SparseDeflate()._hotelling(K, Q[:, count]), Q, count+1

    # Projection deflation
    @staticmethod
    def _projection(K: torch.Tensor, x: torch.Tensor):
        Kx = torch.matmul(K, x)
        xKx = torch.inner(x, Kx)
        x_Kx = torch.outer(x, Kx)
        K -= x_Kx
        K -= torch.t(x_Kx)
        K += torch.outer(x, x) * xKx
        return K

    @staticmethod  # when data matrix is given instead of covariance matrix
    def _projection_data(data: torch.Tensor, x: torch.Tensor):
        data -= torch.mm(data, torch.outer(x, x))
        return data

    # Orthogonalized projection deflation
    @staticmethod
    def _ortho_projection(K: torch.Tensor, x: torch.Tensor, count: int,
                         Q: torch.Tensor):
        # standard Hotelling's deflation for the first round
        if count == 0:
            Q[:, count] = x
            return SparseDeflate()._projection(K, x)

        # OHD for the subsequent round
        else:
            # Gram-Schmidt process
            Q[:, count] = x - torch.matmul(
                torch.mm(Q[:, 0:count], torch.t(Q[:, 0:count])),
                x)
            Q[:, count] /= torch.norm(Q[:, count])
            return SparseDeflate()._projection(K, Q[:, count]), Q, count+1

    # Schur complement deflation, equivalent to orthogonalized Schur complement deflation
    @staticmethod
    def _schur(K: torch.Tensor, x: torch.Tensor):
        Kx = torch.matmul(K, x)
        K -= torch.outer(Kx, Kx) / torch.inner(x, Kx)
        return K

    @staticmethod  # when data matrix is given instead of covariance matrix
    def _schur_data(data: torch.Tensor, x: torch.Tensor):
        Xz = torch.matmul(data, x)
        data -= torch.mm(torch.outer(Xz, Xz) / torch.inner(Xz, Xz), data)
        return data

# endregion
