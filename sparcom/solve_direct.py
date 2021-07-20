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
        l_i = (i % N)
        k_i = np.floor(i/N).astype(int)

        Hz = Mat(H.conj().T @ Z[:,i])
        B = N * fft.ifft(Hz, N, axis=0) # ifft column-wise is same as F^H @ HZ
        u = fft.fft(B[l_i,:].conj().T, N, axis=0)
        v[i] = u[k_i]

    return Vec(v)

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

def calc_Lf(U, N):
    B = calc_B(U, N)
    return np.max(B)

def tau(x, alpha):
    return np.maximum(np.abs(x) - alpha, 0) * np.sign(x)

def solve_direct(f, P, sigma, lamb, kmax):
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
    F_M = dft(N)[:M, :]

    # Vectorize
    Y = Y.reshape(T, M*M)
    Y = np.moveaxis(Y, 0, -1)
    H = np.diag(U.flatten())

    # Create M matrix (SLOW)
    A = H @ np.kron(F_M, F_M)
    Mm = np.square(np.abs(A.conj().T @ A))

    # Get Lipschitz Constant (SLOW)
    Lf = norm(Mm, ord=2)

    # Calculate Empirical Autocovariance
    EY = np.mean(Y, axis=1).reshape(-1, 1)
    R = 1/T * (Y - EY) @ (Y - EY).conj().T

    # Calculate v vector
    V = A.conj().T @ R @ A
    v = np.diag(V.real).reshape(-1, 1)

    # Algorithm 4.1 Fast Proximal Graident Descent
    w = np.ones(v.shape).astype(float)
    x_prev = np.zeros(v.shape).astype(float)
    x = np.zeros(v.shape).astype(float)
    t = t_prev = 1.0
    eps = 1e-4
    itert = trange(kmax, desc='FISTA', leave=True)
    for k in itert:
        # Step 1: Update z
        z = x + ((t_prev - 1)/t) * (x - x_prev)

        # Step 2: Calculate gradient
        grad_f = Mm @ z - v

        # Step 3: Soft-thresholding
        x_prev = x
        x = tau(z - (1/Lf * grad_f), lamb/Lf * w)

        # Step 4: Non-negative orthant projection
        x[x < 0] = 0.0

        # Step 5: Update t
        t_prev = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_prev**2))

        # Step 6: Update weights (only update later on)
        #w = 1/(np.abs(x) + eps)

    # Create final SPARCOM image
    img = np.abs(x.reshape(N, N))
    
    return img
