#!/usr/bin/env python3
"""
Pre-processing to produce axial slice Numpy files
from the input NiFTi format volumes.

Run with --help to see command line options.
"""
import sys, os
import numpy as np
import glob
from tqdm import tqdm
from kits_volume_utils import *

KITS_DIR = "/data/biomedical-imaging/kits19/data"
CACHE_DIR = "img_cache"

def case_from_pid(pid):
    return "case_{:05d}".format(int(pid))


def pid_from_case(case):
    return int(case.split("_")[1])


def main(args=None):

    if args is None:
        kits_base_dir = KITS_DIR
        cache_dir = CACHE_DIR
    else:
        kits_base_dir = args.base_dir
        cache_dir = args.cache_dir

    seg_files = glob.glob(os.path.join(kits_base_dir, "*/segmentation.nii.gz"))
    # print(seg_files[:5])
    # Now extract the case info, and then the PID
    cases_with_segs = [os.path.dirname(f).split(os.path.sep)[-1] for f in seg_files]
    # print(cases_with_segs[:5])
    pids_with_segs = sorted([pid_from_case(c) for c in cases_with_segs])
    # print(pids2019-07-18_with_segs[:5])

    # For each case that has a seg, convert the nifti to a series of slices and save into the following directory structure:
    #
    #     output_dir/
    #        vol/     <- image slices go here
    #        seg/     <- seg slices go here

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "vol"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "seg"), exist_ok=True)

    def check_existing_match(img, path):
        match = False
        if os.path.isfile(path):
            try:
                c_img = np.load(path)
                match = np.all(img == c_img)
            except:
                pass
        return match

    seg_dir = os.path.join(cache_dir, "seg")
    img_dir = os.path.join(cache_dir, "vol")

    for pid in tqdm(pids_with_segs):
        img_vol = load_nifti_volume(pid, kits_base_dir)
        seg_vol = load_nifti_volume(pid, kits_base_dir, is_mask=True)
        for slc_idx in tqdm(range(img_vol.shape[0])):
            img = img_vol[slc_idx, ...]
            seg = seg_vol[slc_idx, ...]
            fname = "{}_{}.npy".format(pid, slc_idx)
            if not check_existing_match(img, os.path.join(img_dir, fname)):
                # print("Caching image {}".format(fname))
                np.save(os.path.join(img_dir, fname), img, allow_pickle=False)
            # else:
            # print("Cache hit for image {}".format(fname))
            if not check_existing_match(seg, os.path.join(seg_dir, fname)):
                # print("Caching mask {}".format(fname))
                np.save(os.path.join(seg_dir, fname), seg, allow_pickle=False)
            # else:
            # print("Cache hit for mask {}".format(fname))


def get_args():
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(prog="{0}".format(os.path.basename(sys.argv[0])))
    ap.add_argument(
        "--base-dir",
        type=str,
        required=False,
        default="data",
        help="Provide path to existing KiTS19-format NiFTi data directory.",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        required=False,
        default="img_cache",
        help="Provide path to the directory where the slice cache will be built.",
    )

    return ap.parse_args()


if __name__ == "__main__":
    main(get_args())
