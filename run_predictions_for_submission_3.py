#!/usr/bin/env python3
"""
Runs predictions for the second scored submission "score2".
"""
import numpy as np
import pandas as pd
import sys, os
from orchestra import *
from predict_multiview import *
from post_processing import *


BASE_DIR = "/scratch/jcausey/biomedical-imaging/kits19/data"


# Syntax for predict_multiview is:
#   predict_multiview(img_vol, weights_file, orientation='axial', tu_thresh=0.1, k_thresh=0.2)
DEFAULT_ENSEMBLE = [
    lambda img_vol: predict_multiview(
        img_vol, weights_file="ensemble_weights/unet_axial_e150.h5"
    ),
    lambda img_vol: predict_multiview(
        img_vol, weights_file="ensemble_weights/unet_axial_KT-T_e205.h5"
    ),
]

DEFAULT_COEFS = [(1, 1), (1, 1)]

DEFAULT_POST_PROC = lambda seg: post_process_kt_t(
    seg,
    k_steps=[],
    t_steps=[
        lambda seg: fill_large_gaps(seg, axis="all"),
        lambda seg: fill_objects(seg, axis="all"),
        lambda seg: keep_largest(seg, n=5),
    ],
    kt_steps=[lambda seg: keep_largest(seg)],
    do_intersect=True,
)


def do_prediction(
    patient_list, data_dir="data", output_dir="predictions", orchestra=None
):
    ## Orchestra:
    # Orchestra(models, weights, datapath, outpath, unity_weights=True)
    if orchestra is not None:
        ensemble = orchestra
    else:
        ensemble = Orchestra(
            models=DEFAULT_ENSEMBLE,
            weights=DEFAULT_COEFS,
            datapath=data_dir,
            outpath=output_dir,
            post_process=DEFAULT_POST_PROC,
        )

    ensemble.predict(patient_list)


def get_args():
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(prog="{0}".format(os.path.basename(sys.argv[0])))
    ap.add_argument(
        "--prediction-list",
        dest="prediction_list",
        type=str,
        required=False,
        default="final_test_list.csv",
        help="Text file containing list of samples for inference.",
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        required=False,
        default=BASE_DIR,
        help="Path to the directory containing the input images.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="predictions",
        help="Name of directory where predictions will be saved.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    patient_list = pd.read_csv(args.prediction_list, header=None).values[:, 0]
    do_prediction(patient_list, args.data_dir, args.output_dir)
