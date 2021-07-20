#!/usr/bin/env python3
import numpy as np
from numpy.linalg import norm

def gauss_kern(N, P=1, sigma=1.0, normalized=False, truncate=4.0):
    '''
    2D Gaussian Kernel with Covariance matrix sigma^2 * I
    '''
    #xx = np.linspace(-N/2, N/2, N)
    xx = np.linspace(-N//2+1, N//2, N//P)
    Xx, Xy = np.meshgrid(xx, xx)
    r = np.vstack([Xx.reshape(1, -1), Xy.reshape(1, -1)])
    kern = np.exp(-0.5 * np.square(norm(r, axis=0)) / sigma**2)
    if truncate is not None:
        kern[kern < np.exp(-0.5 * truncate**2)] = 0.0
    if normalized:
        kern = np.divide(kern, 2*np.pi*sigma)
    return kern.reshape(N//P, N//P)

# define 2D gaussian filter function to mimic 
# MATLAB's fspecial('gaussian', [shape], [sigma])
def matlab_style_gauss_kern(shape=(7,7), sigma=1, normalized=False):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x**2 + y**2) / (2.0*sigma**2))
    h.astype('float32')
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0 and normalized:
        h /= sumh
    if normalized:
        h = h * 2.0
    h = h.astype('float32')
    return h

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(gauss_kern(64, sigma=3.0))
    plt.subplot(122)
    plt.imshow(gauss_kern(64, P=4, sigma=3.0))
    plt.show()
