#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy import fft
from scipy.linalg import dft
from scipy.linalg import norm
from tqdm import trange
from util import gauss_kern

def abs2(arr):
    return np.square(np.abs(arr))

def Vec(X):
    return np.reshape(X, (-1, 1),  order='F')

def Mat(x):
    p = int(np.sqrt(x.size))
    return np.reshape(x, (p, p), order='F')

def relu(x):
    return x * (x > 0.0)

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

def solve_ista(f, P, sigma, lamb, kmax):
    # Get grid sizes and sequence number
    T, M, _ = f.shape
    N = int(M * P)

    # Get the Gaussian kernel
    #xx = np.linspace(-N//2, N//2+1, N)
    #Xx, Xy = np.meshgrid(xx, xx)
    #r = np.vstack([Xx.reshape(1, -1), Xy.reshape(1, -1)])
    #g = gauss_kern(r, sigma)
    #g = g.reshape(N, N)[::P, ::P]
    g = loadmat("data/psf.mat")["psf"]

    # Calculate FFT
    Y = fft.fft2(f)
    U = fft.fft2(fft.ifftshift(g))

    # Vectorize
    Y = Y.reshape(T, M*M)
    Y = np.moveaxis(Y, 0, -1)
    H = np.diag(U.flatten())

    # Calculate Empirical Autocovariance
    EY = np.mean(Y, axis=1).reshape(-1, 1)
    R = 1/T * (Y - EY) @ (Y - EY).conj().T

    # Calculate v (needed for PGD)
    v = calc_v(H, R, M, N)

    # Get Lipschitz Constant
    B = calc_B(U, N)
    Lf = np.max(B).real

    # Solve with ISTA
    x = np.zeros(v.shape).astype(float)
    itert = trange(kmax, desc='ISTA', leave=True)
    for k in itert:
        f = 1/Lf * (calc_Mx(B, x, N) - v)
        x = relu(x - f - lamb/Lf)
        x[x < 0] = 0.0

    # Create final SPARCOM image
    img = np.abs(x.reshape(N, N))
    
    return img
