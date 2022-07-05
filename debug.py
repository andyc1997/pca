import numpy as np
import pandas as pd
import torch

from model_sparse import SPCATPower

# DRAFT ONLY for development.


# region read iris data
X_tr = pd.read_csv(r'.\iris.csv', sep=',')
X_tr = X_tr.drop([X_tr.columns[0], X_tr.columns[5]], axis=1)
print(f'Shape of training data: {X_tr.shape} \n'
      f'Data types: \n{X_tr.dtypes}\n')
# endregion


# region preprocessing
X_tr = np.array((X_tr - X_tr.mean()) / X_tr.std())
print(f'Shape of training data: {X_tr.shape} \n'
      f'Data types: \n{X_tr.dtype}\n')

K = (X_tr.transpose() @ X_tr) / (150 - 1)
K = torch.tensor(K)
print(f'Shape of Gram matrix: {K.shape} \n'
      f'Data types: \n{K.dtype}\n')
# endregion


# region read journee simulated data
X_tr = pd.read_csv(r'.\samples.csv', sep=',', header=None)
print(f'Shape of training data: {X_tr.shape} \n')

X_tr = torch.Tensor(X_tr.to_numpy()).to(torch.float64)
# endregion


# region debug
model = SPCATPower(500, k=2, card=[0.5, 0.5])
Z = model.fit(None, data=X_tr)
print(f'Sparse loading Z: \n{torch.round(Z[:20, :], decimals=4)}\n')
# endregion

