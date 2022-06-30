import numpy as np
import pandas as pd
import torch

from model_sparse import SPCABlockl0


# DRAFT ONLY for development.


#region read iris data
X_tr = pd.read_csv(r'.\iris.csv', sep=',')
X_tr = X_tr.drop([X_tr.columns[0], X_tr.columns[5]], axis=1)
print(f'Shape of training data: {X_tr.shape} \n'
      f'Data types: \n{X_tr.dtypes}\n')
#endregion


#region preprocessing
X_tr = np.array((X_tr - X_tr.mean())/X_tr.std())
print(f'Shape of training data: {X_tr.shape} \n'
      f'Data types: \n{X_tr.dtype}\n')

K = (X_tr.transpose() @ X_tr)/(150 - 1)
K = torch.tensor(K)
print(f'Shape of Gram matrix: {K.shape} \n'
      f'Data types: \n{K.dtype}\n')
#endregion


#region debug
model = SPCABlockl0(4, gamma=[0.1, 0.5, 0.8], mu=[1, 1, 1], k=3)
Z = model.fit(K, trace=True)
print(f'Sparse loading Z: \n{Z}\n')
#endregion