#!/usr/bin/env python3
import os
import argparse
from tifffile import TiffFile, TiffSequence, TiffWriter, imsave
from tqdm import trange

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tifffile', type=str, default="data/sequence.tif", \
            help="path to input .tiff sequence.")
    args = parser.parse_args()

    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/sequence"):
        os.makedirs("data/sequence")

    with TiffFile(args.tifffile) as tif:
        sequence = tif.asarray(key=range(len(tif.pages)))
        print(sequence.shape)

    for i in trange(sequence.shape[0]):
        imsave(f"data/sequence/{i}.tif", sequence[i])

