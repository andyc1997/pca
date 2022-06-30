import torch
import numpy

from model import PCAobj
from torch import linalg
from torch.nn import functional as F
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import SparsePCA

#region [sparse PCA with Elastic Net] Zou, H., Hastie, T., & Tibshirani, R. (2006).
class SPCAENobj(PCAobj):
    # sPCA stands for sparse PCA
    # EN stands for elastic net approach
    def __init__(self, dim:int, l2_lambda:float, l1_lambdas:list, k:int=None):
        super(SPCAENobj, self).__init__(dim, k)
        assert l2_lambda > 0
        assert all([l1_lambda > 0 for l1_lambda in l1_lambdas])
        assert len(l1_lambdas) == self.k

        self.l2_lambda = l2_lambda
        self.l1_lambdas = l1_lambdas

    def fit(self, K:torch.Tensor, max_iter:int=5000, tol:float=1e-4):
        # perform classical PCA on K
        super(SPCAENobj, self).fit(K)

        # initialize parameters
        A = self.V_topk.clone()
        B = torch.zeros((self.dim, self.k), dtype=torch.float64)
        X = self.get_pseudodata()
        A_prev, B_prev = None, None

        # alternate algorithm
        iter = 1
        while iter < max_iter:
            # save previous status
            A_prev, B_prev = A.clone(), B.clone()

            # solve a sequence of elastic net problems
            # update B
            for j in range(self.k):
                # get parameters
                Y = torch.matmul(X, A[:, j])
                l1_lambda = self.l1_lambdas[j]

                # get beta estimates. implementation in sklearn uses coordinate algorithm
                alpha = 2*self.l2_lambda + l1_lambda
                l1_ratio = l1_lambda/(l1_lambda + 2*self.l2_lambda)
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False,
                                   precompute=K.numpy())
                model.fit(X.numpy(), Y.numpy())
                B[:, j] = torch.Tensor(model.coef_)

            # update A, t(X)*X*B might not be symmetric
            U, _, Vh = linalg.svd(torch.mm(K, B), full_matrices=False)
            A = torch.mm(U, Vh)

            # convergence
            if torch.max(torch.abs(A - A_prev)) < tol and torch.max(torch.abs(B - B_prev)) < tol:
                break

            iter += 1

        # report iterations
        if iter == max_iter:
            print(f'\nMaximum iteration exceeds: {max_iter}.\n'
                  f'\nMaximum absolute difference in A: {torch.max(torch.abs(A - A_prev))}.\n'
                  f'\nMaximum absolute difference in B: {torch.max(torch.abs(B - B_prev))}.\n')
        else:
            print(f'\nNumber of iterations: {iter}. Convergence achieved.\n')

        # normalize each column of B by 2-norm
        B = F.normalize(B, dim=0)
        return B
#endregion


#region [sparse PCA with L1 Sparse Coding] Mairal, J., Bach, F., Ponce, J., & Sapiro, G. (2009, June).
class SPCAL1obj(PCAobj):
    # sPCA stands for sparse PCA
    # L1 stands for L1 sparse coding approach
    # as implemented in sklearn
    def __init__(self, dim:int, alpha:float=1, ridge_alpha:list=0.01, k:int=None):
        super(SPCAL1obj, self).__init__(dim, k)
        assert ridge_alpha > 0, f'Positive alpha should be passed, but got {ridge_alpha}\n'
        assert alpha > 0, f'Positive alpha should be passed, but got {alpha}\n'

        self.alpha = alpha
        self.ridge_alpha = ridge_alpha

    def fit(self, K:torch.Tensor, max_iter:int=1000, tol:float=1e-6, method:str='lars'):
        # perform classical PCA on K
        super(SPCAL1obj, self).fit(K)

        # initialize parameters
        X = self.get_pseudodata().numpy()
        model = SparsePCA(n_components=self.k, alpha=self.alpha, ridge_alpha=self.ridge_alpha,
                          max_iter=max_iter, tol=tol, method=method)

        # algorithm
        model.fit(X)
        return torch.t(torch.Tensor(model.components_))
#endregion


#region [block sparse PCA with GPower l0 penalty] Journée, M., Nesterov, Y., Richtárik, P., & Sepulchre, R. (2010).
class SPCABlockl0(PCAobj):
    # sPCA stands for sparse PCA
    # Block l0 stands for L0 penalty in block
    def __init__(self, dim:int, gamma:list=None, mu:list=None, k:int=None,
                 max_iter: int = 1000, tol: float = 1e-4, trace: bool = True):
        super(SPCABlockl0, self).__init__(dim, k)

        # validation for hyperparameters
        assert gamma is not None and mu is not None, f'hyperparameters should be provided\n'
        assert len(gamma) == k, f'invalid length for gamma, got: {len(gamma)}\n'
        assert len(mu) == k, f'invalid length for mu, got: {len(mu)}\n'
        assert all([g > 0 and g <= 1 for g in gamma]), f'invalid range for gamma, got: {gamma}\n'
        assert all([m > 0 for m in mu]), f'invalid range for mu, got: {mu}\n'

        self.gamma = torch.Tensor(gamma).double()
        self.mu = torch.Tensor(mu).double()
        self.max_iter = max_iter
        self.tol = tol
        self.trace = trace

    def fit(self, K:torch.Tensor, data:torch.Tensor=None):
        # perform classical PCA on K
        super(SPCABlockl0, self).fit(K)

        # Check if data is provided
        A = data
        if data is None and K is not None:
            A = self.get_pseudodata() # where t(A)*A = K

        # initialize parameters
        idx_max = torch.argmax(torch.norm(A, dim=0))
        a_norm_max = torch.norm(A[:, idx_max]) # should be 1 if K is provided

        # initialize X by QR of [column of A with max norm | randn matrix], recommended in the paper
        X_init = torch.randn((self.dim, self.k), dtype=torch.float64)
        X_init[:, 0] = A[:, idx_max]/a_norm_max
        X, R = linalg.qr(X_init)
        gamma = self.gamma * torch.square(torch.matmul(torch.t(R), self.mu))  # rescale sparsity factor

        # cost
        cost, cost_prev = 0.0, None

        # algorithm
        iter = 1
        while iter < self.max_iter:
            # save previous status
            cost_prev = cost

            # compute threshold
            dot_prod_ax = torch.mm(torch.t(A), X)
            dot_prod_ax *= self.mu.repeat((self.dim, 1))
            tresh = torch.clamp(torch.square(dot_prod_ax) - gamma.repeat((self.dim, 1)), 0)

            # update cost, terminate if it converged
            cost = torch.sum(tresh)
            if torch.abs(cost) < 1e-13:
                break

            # compute gradient
            else:
                grad = torch.zeros((self.dim, self.k), dtype=torch.float64)

                for j in range(self.k):
                    pattern = tresh[:, j] > 0.0
                    if torch.sum(pattern) > 1: # matrix-vector product
                        grad[:, j] = torch.matmul(A[:, pattern], dot_prod_ax[pattern, j])

                    elif torch.sum(pattern) == 1: # vector-scalar product
                        grad[:, j] = A[:, pattern].flatten() * dot_prod_ax[pattern, j]

                    # otherwise, pattern has all False. do nothing. scale gradient for all cases
                    grad[:, j] *= self.mu[j]

            # polar decomposition by SVD
            U, _, Vh = linalg.svd(grad, full_matrices=False)
            X = torch.mm(U, Vh)

            # convergence condition
            if iter > 1 and (cost - cost_prev)/cost_prev < self.tol:
                break

            iter += 1

        # report iterations
        if self.trace:
            if iter >= self.max_iter:
                print(f'Maximum iteration exceeds: {self.max_iter}.\n'
                      f'Relative cost difference: {(cost - cost_prev)/cost_prev}.\n')
            elif torch.abs(cost) < 1e-13:
                print(f'Sparsity factors are too high.\n')
            else:
                print(f'Number of iterations: {iter}. Convergence achieved.\n')

        # locally optimal sparsity pattern
        P = torch.where(torch.square(torch.matmul(torch.t(A), X * self.mu.repeat((self.dim, 1)))) >
                        gamma.repeat((self.dim, 1)), 1, 0)
        P_inv = P == 0

        # postprocessing for sparse loading
        Z = torch.mm(torch.t(A), X)
        Z[P_inv] = 0.0
        Z /= torch.norm(Z, dim=0)

        # handle nan values in loading
        if torch.any(torch.isnan(Z)): print('Nan found in sparse loading\n')
        Z = torch.nan_to_num(Z)
        return Z
#endregion


#region [Single unit sparse PCA with GPower l0 penalty] Journée, M., Nesterov, Y., Richtárik, P., & Sepulchre, R. (2010).
class SPCASingleUnitl0(PCAobj):
    # sPCA stands for sparse PCA
    # SingleUnitl0 stands for L0 penalty and components are extracted unit by unit
    def __init__(self, dim:int, gamma:list=None, k:int=None,
                 max_iter: int = 1000, tol: float = 1e-4, trace: bool = True):
        super(SPCASingleUnitl0, self).__init__(dim, k)

        # validation for hyperparameters
        assert gamma is not None, f'hyperparameters should be provided\n'
        assert len(gamma) == k, f'invalid length for gamma, got: {len(gamma)}\n'
        assert all([g > 0 and g <= 1 for g in gamma]), f'invalid range for gamma, got: {gamma}\n'

        self.gamma = torch.Tensor(gamma).double()
        self.max_iter = max_iter
        self.tol = tol
        self.trace = trace

    def fit(self, K:torch.Tensor, data:torch.Tensor=None):
        # perform classical PCA on K
        super(SPCASingleUnitl0, self).fit(K)