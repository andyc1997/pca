import numpy as np
import pandas as pd
import torch

from model import VarExplain
from model_sparse import SPCATPower, SPCASingleUnitl0

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
K = torch.mm(torch.t(X_tr), X_tr)
# endregion


# region debug
model = SPCATPower(500, k=5, card=[0.02, 0.02, 0.02, 0.02, 0.02], max_iter=1000)
# model = SPCASingleUnitl0(500, gamma=0.3, k=5)
Z = model.fit(K)
print(f'Sparse loading Z: \n{torch.round(Z[:20, :], decimals=4)}\n')

varexplain = VarExplain(k=5)
varexplain.fit(X_tr, Z)
print(varexplain.cvar)
# endregion

