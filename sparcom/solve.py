#!/usr/bin/env python3
import numpy as np
import numpy.fft as fft
from scipy.linalg import dft
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import trange
from util import gauss_kern

def abs2(arr):
    return np.square(np.abs(arr))

def Vec(X):
    return np.reshape(X, (-1, 1),  order='F')

def Mat(x):
    p = int(np.sqrt(x.size))
    return np.reshape(x, (p, p), order='F')

# Algorithm 5.2 from SPARCOM Paper
def calc_v(H, R, M, N):
    Q = H.conj().T @ R
    Z = np.zeros([N*N, M*M], dtype=complex)
    v = np.zeros(N*N, dtype=float)
    t1 = trange(M*M, desc='Calculating Z^H', leave=True)
    for i in t1:
        q = Q[:,i]
        Ti = N * fft.ifft(Mat(q), N, axis=0)
        Ei = fft.fft(Ti.conj().T, N, axis=0)
        Z[:,i] = Vec(Ei.conj().T)[:,0]
    
    Z = Z.conj().T
    t2 = trange(N*N, desc='Calculating v', leave=True)
    for i in t2:
        # special indicies
        l_i = i % N
        k_i = i // N

        Hz = Mat(H.conj().T @ Z[:,i])
        B = N * fft.ifft(Hz, N, axis=0) # ifft column-wise is same as F^H @ HZ
        u = fft.fft(B[l_i,:].conj().T, N, axis=0)
        v[i] = u[k_i].real

    return Vec(v).real

# Algorithm 5.1 from SPARCOM Paper
def calc_Mx(B, x, N):
    X = x.reshape(B.shape, order='F') # fortran-order important for this operation
    Q = B * fft.fft2(X)
    Y = fft.ifft2(Q)
    Mx = Vec(Y)
    return Mx.real

def calc_B(U, N):
    T = N * fft.ifft(np.square(np.abs(U)), N, axis=0)
    E = fft.fft(T.conj().T, N, axis=0)
    B = fft.fft2(np.square(np.abs(E.conj().T)))
    return B

def tau(x, alpha):
    return np.maximum(np.abs(x) - alpha, 0) * np.sign(x)

def solve(f, P, sigma, lamb, kmax, g=None):
    T, M, _ = f.shape
    N = int(M * P)

    # Get kernel of same shape as input
    #xaxis = np.linspace(-N//2, N//2+1, N)
    #Xax, Yax = np.meshgrid(xaxis, xaxis)
    #r = np.vstack((Xax.reshape(1,-1), Yax.reshape(1,-1)))
    #g = gauss_kern(r, sigma)
    #g = g.reshape(N, N)
    if g is None:
        g = loadmat("data/psf.mat")["psf"]
        g = g / np.max(g)
    print(f"g: {g.dtype}, {g.shape}, {np.max(g)}, {np.min(g)}")

    # Calculate FFT
    Y = fft.fft2(f, axes=(-2,-1))
    U = fft.fft2(fft.ifftshift(g))

    # Vectorize
    Y = np.moveaxis(Y, 0, -1)
    Y = Y.reshape(M*M, T, order='F')
    H = np.diag(U.flatten(order='F'))

    # Calculate Empirical Autocovariance
    EY = np.mean(Y, axis=1).reshape(-1,1, order='F')
    Y  = Y - EY
    R = 1/T * Y @ Y.conj().T

    # Calculate v (needed for PGD)
    v = calc_v(H, R, M, N)

    # Calculate Lf
    B = calc_B(U, N)
    Lf = np.max(B).real

    # Initialize
    x = np.zeros(v.shape, dtype=float)
    x_prev = np.zeros(v.shape, dtype=float)
    w = np.ones(v.shape, dtype=float)
    t = t_prev =  1.0
    eps = 1e-4

    # Algorithm 4.1 Fast Proximal Graident Descent
    itertout = trange(4, desc='Reweight', leave=True)
    itert = trange(kmax, desc='FISTA', leave=True)
    for p in itertout:
        for k in itert:
            # Step 1: Update z
            z = x + ((t_prev - 1)/t) * (x - x_prev)

            # Step 2: Calculate Gradient (using efficient method)
            Mz_k = calc_Mx(B, z, N)
            grad_f = Mz_k - v

            # Step 3: Soft-thresholding
            x_prev = x
            x = tau(z - (1/Lf * grad_f), lamb/Lf * w)

            # Step 4: Non-negative orthant projection
            x[x < 0] = 0.0

            # Step 5: Update t
            t_prev = t
            t = 0.5 * (1 + np.sqrt(1 + 4*t_prev**2))

        # Update weights
        w = 1/(np.abs(x) + eps)


    # Create final SPARCOM image
    #img = np.flipud(np.abs(x.reshape(N, N)))
    img = np.abs(x.reshape(N, N), order='F')
    return img

def solve_patch(f, P, sigma, lamb, kmax):
    pass
