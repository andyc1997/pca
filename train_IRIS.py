import numpy as np
import pandas as pd
import torch

from model import PCAobj


#region read wine data
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


#region perform PCA
model = PCAobj(dim=4)
model.fit(K)
print(f'Stdev: \n{torch.sqrt(model.s)}\n')
print(f'Eigenvectors: \n{model.V}\n')
#endregion


#region correct output
# Loadings:
#              Comp.1  Comp.2  Comp.3  Comp.4
# Sepal.Length  0.5211  0.3774  0.7196  0.2613
# Sepal.Width  -0.2693  0.9233 -0.2444 -0.1235
# Petal.Length  0.5804  0.0245 -0.1421 -0.8014
# Petal.Width   0.5649  0.0669 -0.6343  0.5236
#
# Stedv:
# Comp.1 Comp.2 Comp.3 Comp.4
# 1.7084 0.9560 0.3831 0.1439
#endregion