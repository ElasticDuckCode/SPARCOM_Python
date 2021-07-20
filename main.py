#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile, TiffSequence, TiffWriter
import sparcom
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SPARCOM input parameters
    parser.add_argument('--tifffile', type=str, default="data/sequence.tif", \
            help="path to input 32-bit floating point .tiff sequence.")
    parser.add_argument('--truth', type=str, default="data/truth.tif", \
            help="path to ground truth 32-bit floating point .tiff image (for validation).")
    parser.add_argument('--kern_var', type=float, default=3.0, \
            help="variance of Gaussin kernel in input sequence.")
    parser.add_argument('--scale', type=int, default=4, \
            help="upscaling factor of final image.")

    # SPARCOM PGD parameters
    parser.add_argument('--lamb', type=float, default=1e-10, \
            help="lambda value for proximal gradient descent")
    parser.add_argument('--kmax', type=float, default=2000, \
            help="max number of proximal gradient descent iterations.")
    args = parser.parse_args()

    with TiffFile(args.truth) as tif:
        truth = tif.asarray()
        print(f"truth: {truth.shape}")
    with TiffFile(args.tifffile) as tif:
        sequence = tif.asarray(key=range(len(tif.pages))) * 255.0
        sequence = sequence - np.median(sequence, axis=0)
        #sequence = (sequence.astype(float).T / np.max(sequence, axis=(1,2))).T
        print(f"sequence: {sequence.shape}, {sequence.dtype}, {np.max(sequence)}, {np.min(sequence)}")

    avg = np.mean(sequence, axis=0)
    print(f"avg: {avg.shape}, {avg.dtype}, {np.max(avg)}, {np.min(avg)}")


    plt.subplot(131)
    plt.imshow(avg)
    plt.subplot(132)
    pred = sparcom.solve(sequence, args.scale, args.kern_var, args.lamb, args.kmax)
    #pred = sparcom.solve_direct(sequence, args.scale, args.kern_var, args.lamb, args.kmax)
    #pred = sparcom.solve_ista(sequence, args.scale, args.kern_var, args.lamb, args.kmax)
    plt.imshow(pred)
    plt.subplot(133)
    plt.imshow(truth)
    plt.show()




