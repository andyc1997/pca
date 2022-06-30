import numpy as np

# region reference

# How to calculate a Gaussian kernel matrix efficiently in numpy?
# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

#endregion

def gkern(l=20, sig=5.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

kmat = gkern()
_, s, _ = np.linalg.svd(kmat)
