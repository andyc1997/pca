import numpy as np

# H. Shen, J.Z. Huang / Journal of Multivariate Analysis 99 (2008) 1015 – 1034 1023
# 3.1. Data generation from a sparse PCA model


# region initialize parameters
n = 500  # number of features
m = 2  # number of sparse orthonormal vectors
p = 50  # number of samples
# endregion


# region covariance structures
eigval = np.ones(n)
eigval[0] = 400
eigval[1] = 300
eigval = np.diag(eigval)

eigvec = np.zeros((n, n))
eigvec[:10, 0] = 1 / np.sqrt(10)
eigvec[10:20, 1] = 1 / np.sqrt(10)
eigvec[:, m:] = np.random.rand(n, n - m)
# endregion


# region rank check
rank = np.linalg.matrix_rank(eigvec)
while rank < n:
    eigvec[:, m:] = np.random.rand(n, n - m)
    rank = np.linalg.matrix_rank(eigvec)
# endregion


# region simulation
eigvec, _ = np.linalg.qr(eigvec)
K = (eigvec @ eigval) @ eigvec.T

N_sim = 500 # number of sample points
X_sim = eigvec @ (np.sqrt(eigval) @ np.random.randn(N_sim, n).T)
X_sim = X_sim.T
# endregion

np.savetxt('cov_mat.csv', K, delimiter=',')
np.savetxt('samples.csv', X_sim, delimiter=',')


