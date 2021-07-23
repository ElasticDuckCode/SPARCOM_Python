#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile, TiffSequence, TiffWriter
from scipy.io import loadmat
import tifffile
import sparcom
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SPARCOM input parameters
    parser.add_argument('--tifffile', type=str, default="data/sequence.tif", \
            help="path to input 32-bit floating point .tiff sequence.")
    parser.add_argument('--truth', type=str, default="data/truth.tif", \
            help="path to ground truth 32-bit floating point .tiff image (for validation).")
    parser.add_argument('--kern_mat', type=str, default="data/psf.mat", \
            help="mat file containing low-resolution sampled psf.")
    parser.add_argument('--scale', type=int, default=4, \
            help="upscaling factor of final image.")

    # SPARCOM PGD parameters
    parser.add_argument('--lamb', type=float, default=1e-10, \
            help="lambda value for proximal gradient descent")
    parser.add_argument('--kmax', type=float, default=1000, \
            help="max number of proximal gradient descent iterations.")
    args = parser.parse_args()

    #with TiffFile(args.truth) as tif:
    #    truth = tif.asarray()
    #    print(f"truth: {truth.shape}")
    with TiffFile(args.tifffile) as tif:
        sequence = tif.asarray(key=range(len(tif.pages))).astype(np.float32)
        #sequence = (sequence.astype(float).T / np.max(sequence, axis=(1,2))).T
        print(f"sequence: {sequence.shape}, {sequence.dtype}, {np.max(sequence)}, {np.min(sequence)}")

    avg = np.mean(sequence, axis=0)
    print(f"avg: {avg.shape}, {avg.dtype}, {np.max(avg)}, {np.min(avg)}")

    # load in psf .mat file
    g = loadmat(args.kern_mat)["psf"]
    #g = g / np.max(g)
    print(f"g: {g.dtype}, {g.shape}, {np.max(g)}, {np.min(g)}")

    # pre-processing for SPARCOM
    sequence /= sequence.max()
    sequence -= np.median(sequence, axis=0)
    sequence *= 255.0

    #pred = sparcom.solve_direct(sequence, args.scale, g, args.lamb, args.kmax)
    #pred = sparcom.solve(sequence, args.scale, g, args.lamb, args.kmax)
    pred = sparcom.solve_ista(sequence, args.scale, g, args.lamb, args.kmax)
    #pred = sparcom.solve_patch(sequence, args.scale, g, args.lamb, args.kmax)

    plt.rcParams['image.cmap'] = 'hot'
    plt.figure(figsize=(8,4), constrained_layout=True)
    plt.subplot(121)
    plt.imshow(avg)
    plt.title("Diffraction Limited")
    plt.subplot(122)
    plt.imshow(pred)
    plt.title("SPARCOM")
    plt.show()
    #plt.close()

    #plt.subplot(133)
    #plt.imshow(truth)

    pred = pred / pred.max()
    avg = avg / avg.max()
    tifffile.imwrite("sparcom.tif", pred.astype(np.float32))
    tifffile.imwrite("avg.tif", avg.astype(np.float32))




