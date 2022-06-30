import numpy as np
import idx2numpy
import torch

from model import PCAobj
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

## MNIST


#region load images into tensor
X_tr = idx2numpy.convert_from_file(r'..\deep-clustering-with-conv-autoencoders\train-images.idx3-ubyte')
X_tst = idx2numpy.convert_from_file(r'..\deep-clustering-with-conv-autoencoders\t10k-images.idx3-ubyte')

(N_tr, nx, ny) = X_tr.shape
(N_tst, _, _) = X_tst.shape

# add single channel and flatten
X_tr = np.expand_dims(X_tr, axis=1).reshape((N_tr, nx*ny))
X_tst = np.expand_dims(X_tst, axis=1).reshape((N_tst, nx*ny))

# normalize into [0, 1]
X_tr = torch.Tensor(X_tr/255.0)
X_tst = torch.Tensor(X_tst/255.0)

dataset_tr, dataset_tst = TensorDataset(X_tr), TensorDataset(X_tst)

batch_size = 64
dataloader_tr = DataLoader(X_tr, batch_size=batch_size)
dataloader_tst = DataLoader(X_tst, batch_size=batch_size)
#endregion

#region debug
dim = 64
k = 10
model = PCAobj(dim=dim, k=k)

for i, x in enumerate(dataloader_tr, start=0):
    if i > 0:
        break

    K = torch.mm(torch.t(x), x)
    print(f'Tensor shape of mini-batch: {x.shape} \n'
          f'Tensor shape of covariance matrix {K.shape} \n')

    model.fit(K)
    print(f'U matrix: {model.U.shape} \n'
          f'V matrix: {model.V.shape}')


#endregion

from model import sPCAENobj