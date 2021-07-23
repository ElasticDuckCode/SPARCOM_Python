#!/usr/bin/env python3
import numpy as np
import numpy.fft as fft
from scipy.linalg import dft
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
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
    #for i in range(M*M):
    for i in trange(M*M):
        q = Q[:,i]
        Ti = N * fft.ifft(Mat(q), N, axis=0)
        Ei = fft.fft(Ti.conj().T, N, axis=0)
        Z[:,i] = Vec(Ei.conj().T)[:,0]
    
    Z = Z.conj().T
    HZ = H.conj().T @ Z
    #for i in range(N*N):
    for i in trange(N*N):
        # special indicies
        l_i = i % N
        k_i = i // N
        Hz = Mat(HZ[:,i])
        B = N * fft.ifft(Hz, N, axis=0) # ifft column-wise is same as F^H @ HZ
        u = fft.fft(B[l_i,:].conj().T, N, axis=0)
        v[i] = u[k_i].real

    return Vec(v).real

# Algorithm 5.1 from SPARCOM Paper
def calc_Mx(B, x, N):
    #X = x.reshape(B.shape, order='F')
    X = x.reshape(B.shape)
    Q = B * fft.fft2(X)
    Y = fft.ifft2(Q)
    #Mx = Vec(Y)
    Mx = Y.reshape(-1,1)
    return Mx.real

def calc_B(U, N):
    T = N * fft.ifft(np.square(np.abs(U)), N, axis=0)
    E = fft.fft(T.conj().T, N, axis=0)
    B = fft.fft2(np.square(np.abs(E.conj().T)))
    return B

def tau(x, alpha):
    return np.maximum(np.abs(x) - alpha, 0) * np.sign(x)

def solve(f, P, g, lamb, kmax, progress=True):
    T, M, _ = f.shape
    N = int(M * P)

    # Calculate FFT
    Y = fft.fftshift(fft.fft2(f))
    U = fft.fftshift(fft.fft2(fft.ifftshift(g)))

    # Vectorize
    Y = Y.reshape(T, M*M)
    Y = np.moveaxis(Y, 0, -1)
    H = np.diag(U.flatten())

    # Calculate Empirical Autocovariance
    EY = np.mean(Y, axis=1).reshape(-1,1)
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
    eps = 1

    # Algorithm 4.1 Fast Proximal Graident Descent
    #itertout = trange(4, desc='Reweighted FISTA', leave=True)
    for _ in range(4):
        for k in range(kmax):
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
    img = np.abs(x.reshape(N, N))
    img = img / img.max()
    return img

def solve_patch(f, P, sigma, lamb, kmax):
    pass
