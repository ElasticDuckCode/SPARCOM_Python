#!/usr/bin/env python3
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from patchify import patchify, unpatchify
from .solve import *

def solve_patch(f, P, g, lamb, kmax):
    
    T, M, _ = f.shape
    N = int(M * P)
    ps = 16
    
    # Extract proper g for patch size
    gg = fft.ifftshift(g)
    g_tl = gg[:ps//2, :ps//2]
    g_tr = gg[:ps//2, -ps//2:]
    g_bl = gg[-ps//2:, :ps//2]
    g_br = gg[-ps//2:, -ps//2:]
    gg = fft.fftshift(np.block([[g_tl, g_tr],[g_bl, g_br]]))

    # Build image patches
    patches = []
    for i in range(T):
        patch = patchify(f[i], (ps, ps), step=ps//2)
        patches.append(patch)
    patches = np.asarray(patches)

    patch_idx = patches.shape[1:3]

    # Process each patch
    patches_hr = np.zeros([*patch_idx, ps*P, ps*P], dtype=np.float32)
    for n, m in np.ndindex(patch_idx):
        print(f"({m}, {n})")
        dataset = patches[:, n, m, :, :]
        patches_hr[n, m, :, :] = solve(dataset, P, gg, lamb, kmax)

    pred = unpatchify(patches_hr, (N, N))
    print(pred.shape)
    #pred = convolve(pred, g)


    return pred
