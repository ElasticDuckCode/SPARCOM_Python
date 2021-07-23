#!/usr/bin/env python3
import os
import numpy as np
import numpy.fft as fft
import tifffile
import argparse
from scipy.ndimage import gaussian_filter, convolve
from scipy.io import savemat
from tqdm import trange
from util import gauss_kern, matlab_style_gauss_kern

def rgb2gray(rgb):
    if len(rgb.shape) > 2:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        return rgb

def flourescentImage(gt, sparse=5):
    '''
    Create image with random flourescence from ground truth.
    '''
    fl = np.array(gt)

    # Turn on only s-sparse sources in image.
    k = np.count_nonzero(gt)
    s = np.zeros(k)
    s[:sparse] = 1.0
    np.random.shuffle(s)
    fl[fl > 0.0] = s

    # determine the source brightness if on
    q = np.count_nonzero(fl)
    fl[fl > 0.0] = np.random.randn(q)
    return fl

def genGroundTruth(size, source_num):
    gt = np.zeros(size, dtype=np.float32) 
    xaxis = np.linspace(0, size[0]-1, size[0], dtype=int)
    yaxis = np.linspace(0, size[1]-1, size[1], dtype=int)
    kx = np.random.choice(xaxis, size=(source_num, 1), replace=False)
    ky = np.random.choice(yaxis, size=(source_num, 1), replace=False)
    k = np.hstack((kx, ky))
    for m, n in k:
        gt[m, n] = 1.0
    return gt

def groundTruthFromFile(fname):
    gt = tifffile.imread(fname)
    return gt

def genPointSrcImage(gt, g, noise_pwr=1, downsample_factor=4):
    fl = flourescentImage(gt)
    #lr = gaussian_filter(fl, kernel_std)[::downsample_factor, ::downsample_factor] 

    lr = convolve(fl, g)[::downsample_factor, ::downsample_factor]
    lr = lr + np.random.normal(loc=0, scale=noise_pwr**2, size=lr.shape)
    lr = np.clip(lr, 0.0, 1.0)
    return lr

'''
    Point-Source Data Generating Program
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=1000, \
            help="number of sequence frames.")
    parser.add_argument('--kernel_std', type=float, default=3.0, \
            help="std. dev. of Gaussian kernel.")
    parser.add_argument('--dsf', type=int, default=4, \
            help="low-resolution downsample factor.")
    parser.add_argument('--stack', type=int, default=1, \
            help="place sequence into single tiff stack.")
    args = parser.parse_args()


    num_images = args.num_images
    kernel_std = args.kernel_std
    downsample_factor = args.dsf
    stack_flag = bool(args.stack)
    print(args.stack)
    imtype = np.float32

    # create ground truth
    #gt = genGroundTruth((64,64), 3)
    #gt = groundTruthFromFile("smile.tif")
    gt = rgb2gray(groundTruthFromFile("smile.tif")).astype(float)

    # create psf kernel
    g = gauss_kern(gt.shape[0], sigma=kernel_std)
    #g = gauss_kern(gt.shape[0], kernel_std, normalized=True)
    #g = matlab_style_gauss_kern((gt.shape[0],gt.shape[0]), kernel_std)

    # create folder to store the images
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/sequence") and not stack_flag:
        os.makedirs("data/sequence")

    # create dataset of ground truth images
    tifffile.imsave("data/truth.tif", gt.astype(imtype))
    for i in trange(num_images):
        lr = genPointSrcImage(gt, g, noise_pwr=0.0, downsample_factor=downsample_factor)
        if stack_flag:
            if i == 0:
                tifffile.imsave("data/sequence.tif", lr.astype(imtype))
            else:
                with tifffile.TiffWriter("data/sequence.tif", append=True) as tif: 
                    tif.save(lr.astype(imtype))
        else:
            tifffile.imsave(f"data/sequence/{i}.tif", lr.astype(imtype))


    # save the psf in a .mat file for offical SPARCOM software
    #psf = {"psf": g[::downsample_factor, ::downsample_factor], "sigma": kernel_std}
    psf = {"psf": gauss_kern(gt.shape[0], P=downsample_factor, sigma=kernel_std), "sigma": kernel_std}
    savemat("data/psf.mat", psf)
    
