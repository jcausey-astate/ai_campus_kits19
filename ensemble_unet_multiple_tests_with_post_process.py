#!/usr/bin/env python3
"""
This script is designed as a sandbox for testing different
ensembling and post-processing steps.
Edit to create the ensembles / weights / post-processing
pipelines you want to try, then run to produce results for each.
"""

import numpy as np
import pandas as pd
import sys, os
import csv
from time import strftime
import matplotlib.pyplot as plt
from orchestra import *
from evaluation import *
from post_processing import *
from predict_multiview import *


base_dir = "kits19/data"


# Syntax for predict_multiview is:
#   predict_multiview(img_vol, weights_file, orientation='axial', t_thresh=0.1, k_thresh=0.2)
models_to_check = [
    # [   # "score1"
    #     lambda img_vol: predict_multiview(
    #         img_vol, weights_file="ensemble_weights/unet_axial_e98.h5"
    #     ),
    #     lambda img_vol: predict_multiview(
    #         img_vol, weights_file="ensemble_weights/unet_axial_KT-T_e200.h5"
    #     ),
    # ]
    # [   # "score2" and "final"
    #     lambda img_vol: predict_multiview(
    #         img_vol, weights_file="ensemble_weights/unet_axial_e150.h5"
    #     ),
    #     lambda img_vol: predict_multiview(
    #         img_vol, weights_file="ensemble_weights/unet_axial_KT-T_e205.h5"
    #     ),
    # ],
]
model_descriptions = [
    # "2axial_score1",
    # "2axial_score2",
    # "2axial_final",
]
coefs_to_check = [
    # [(1, 1), (1, 1)],
    # [(1, 1), (1, 1)],
    # [(1, 1), (1, 1)],
]

post_proc_to_check = [
    # lambda seg: post_process_kt_t( # "score1"
    #     seg,
    #     k_steps=[],
    #     t_steps=[
    #         lambda seg: fill_gaps(seg),
    #         lambda seg: fill_gaps(seg, axis=1),
    #         lambda seg: fill_gaps(seg, axis=2),
    #     ],
    #     kt_steps=[lambda seg: keep_largest(seg)],
    #     do_intersect=True,
    # ),
    # lambda seg: post_process_kt_t( # "score2"
    #     seg,
    #     k_steps=[],
    #     t_steps=[
    #         lambda seg: fill_large_gaps(seg, axis="all"),
    #         lambda seg: fill_objects(seg, axis="all"),
    #         lambda seg: keep_largest(seg, n=5),
    #     ],
    #     kt_steps=[lambda seg: keep_largest(seg)],
    #     do_intersect=True,
    # ),
    # lambda seg: post_process_kt_t_2(  # score2-with-kidney-preserving "keep_largest" pproc + singleslice removal  (final-score)
    #     seg,
    #     k_steps=[],
    #     t_steps=[
    #         lambda seg: fill_large_gaps(seg, axis="all"),
    #         lambda seg: fill_objects(seg, axis="all"),
    #         lambda seg: remove_single_slice_objects(seg),
    #         lambda seg: keep_largest(seg, n=5),
    #     ],
    #     kt_steps=[lambda seg, vol_kt: keep_largest_intersecting_K(seg, vol_kt)],
    #     do_intersect=True,
    # ),
]

def compare_segs(gt_vol, pred_vol):
    return  # Stand-alone -- do not vis. Comment this line to re-enable
    import matplotlib.pyplot as plt

    # Find maximal segmentation slice and use that...
    has_segs = np.argmax([v.sum() for v in gt_vol])
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 6, 1)
    plt.imshow(pred_vol[has_segs].astype(int), cmap="Set1")
    plt.subplot(1, 6, 2)
    plt.imshow((pred_vol[has_segs] == 1).astype(int), cmap="gray")
    plt.subplot(1, 6, 3)
    plt.imshow((pred_vol[has_segs] == 2).astype(int), cmap="gray")
    plt.subplot(1, 6, 4)
    plt.imshow(gt_vol[has_segs].astype(int), cmap="Set1")
    plt.subplot(1, 6, 5)
    plt.imshow((gt_vol[has_segs] == 1).astype(int), cmap="gray")
    plt.subplot(1, 6, 6)
    plt.imshow((gt_vol[has_segs] == 2).astype(int), cmap="gray")
    plt.show()


def check_dice_on_list(
    patient_list,
    vis=True,
    report_after_n=None,
    orchestra=None,
    post_proc=None,
    save_masks=None,
):
    if orchestra is not None:
        ensemble = orchestra

    def print_summary(p_list, dice_list):
        print(
            "Averages for N={} volumes: KT: {:.3f} std {:.3f} TU: {:.3f} std {:.3f}\n".format(
                len(p_list),
                dice_list[:, 1].mean(),
                dice_list[:, 1].std(ddof=1),
                dice_list[:, 2].mean(),
                dice_list[:, 2].std(ddof=1),
            )
        )

    if report_after_n is None:
        report_after_n = []
    elif type(report_after_n) != type([]):
        report_after_n = [report_after_n]
    all_dice = []
    for pid_idx, pid in enumerate(patient_list):
        print(
            "Predicting for case {} ({} of {})".format(
                pid, (pid_idx + 1), len(patient_list)
            )
        )
        dirname = "case_{:05d}".format(pid)
        dirpath = os.path.join(base_dir, dirname)
        img_file = os.path.join(dirpath, "imaging.nii.gz")
        seg_file = os.path.join(dirpath, "segmentation.nii.gz")
        img_vol, affine = ensemble.load_nifti(img_file)
        pred_vol = ensemble.analyze_volume(img_vol)
        if post_proc is not None:
            print("Applying post processing...")
            pred_vol = post_proc(pred_vol)
        del img_vol
        if save_masks is not None:
            pred_file = "prediction_{:05d}.nii.gz".format(pid)
            ensemble.save_mask(pred_file, pred_vol, affine)
        gt_seg, _ = ensemble.load_nifti(seg_file, is_mask=True)
        kt_dice, t_dice = kt_dice_score(gt_seg, pred_vol)
        print("KT Dice: {}, TU Dice: {}".format(kt_dice, t_dice))
        all_dice.append([pid, kt_dice, t_dice])

        if (vis is True) or (type(vis) == type([]) and pid_idx in vis):
            compare_segs(gt_seg, pred_vol)
        del gt_seg
        del pred_vol

        # If we need an early report, do so now:
        if (pid_idx + 1) in report_after_n:
            print_summary(patient_list[: (pid_idx + 1)], np.asarray(all_dice))

    all_dice = np.array(all_dice)

    print_summary(patient_list, all_dice)

    return all_dice


def write_report(fname, score_data):
    """
    Writes avg, stddev of dice scores, then scores for each patient.
    """
    # {'pid':all_dice[:,0],'kt_dice':all_dice[:,1], 't_dice':all_dice[:,2]}
    score_data = np.asarray(score_data)
    kt_avg = score_data[:, 1].mean()
    kt_std = score_data[:, 1].std(ddof=1)
    t_avg = score_data[:, 2].mean()
    t_std = score_data[:, 2].std(ddof=1)
    with open(fname, "w") as fout:
        writer = csv.writer(fout)
        writer.writerow(["", "pid", "kt_dice", "t_dice"])
        writer.writerow(["avg", "*", round(kt_avg, 3), round(t_avg, 3)])
        writer.writerow(["stddev", "*", round(kt_std, 3), round(t_std, 3)])
        for idx, row in enumerate(score_data):
            writer.writerow([idx + 1, row[0], row[1], row[2]])


def run_multiple_scenarios(patient_list, save_masks=None):
    # First with unity weights
    print("\n\nUnity weights.")
    for idx, models in enumerate(models_to_check):
        report_name = "model_{}_unity.csv".format(model_descriptions[idx])
        if not os.path.isfile(report_name):
            coefs = coefs_to_check[idx]
            if post_proc_to_check is not None:
                post_proc = post_proc_to_check[idx]
            else:
                post_proc = None
            ## Orchestra:
            # Orchestra(models, weights, datapath, outpath, unity_weights=True)
            ensemble = Orchestra(
                models, coefs, "", outpath=save_masks if save_masks is not None else ""
            )
            all_dice = check_dice_on_list(
                patient_list,
                vis=False,
                report_after_n=[10],
                orchestra=ensemble,
                post_proc=post_proc,
                save_masks=save_masks,
            )
            write_report(report_name, all_dice)
    # Now with non-unity weights (Uncomment to try this option.)
    # print("\n\nNon-Unity weights.")
    # for idx, models in enumerate(models_to_check):
    #     report_name = "model_set_{}_NONunity.csv".format(idx)
    #     if not os.path.isfile(report_name):
    #         coefs = coefs_to_check[idx]
    #         ## Orchestra:
    #         # Orchestra(models, weights, datapath, outpath, unity_weights=True)
    #         ensemble = Orchestra(models, coefs, "", "", unity_weights=False)
    #         all_dice = check_dice_on_list(patient_list, vis=False, report_after_n=[10], orchestra=ensemble)
    #         write_report("model_set_{}_NONunity.csv".format(idx), all_dice)
    # Now with unity weights but lower threshold:
    # print("\n\nUnity weights, threshold = K=0.20,T=.15")
    # for idx, models in enumerate(models_to_check):
    #     if len(models) < 2:
    #         continue
    #     report_name = "model_set_{}_unity_ktt-20_tut-15_postproc.csv".format(idx)
    #     if not os.path.isfile(report_name):
    #         coefs = coefs_to_check[idx]
    #         post_proc = post_proc_to_check[idx]
    #         ## Orchestra:
    #         # Orchestra(models, weights, datapath, outpath, unity_weights=True)
    #         ensemble = Orchestra(models, coefs, "", "")
    #         ensemble.k_thresh = 0.20
    #         ensemble.t_thresh = 0.15
    #         all_dice = check_dice_on_list(
    #             patient_list,
    #             vis=False,
    #             report_after_n=[10],
    #             orchestra=ensemble,
    #             post_proc=post_proc,
    #         )
    #         write_report(
    #             "model_set_{}_unity_ktt-20_tut-15_postproc.csv".format(idx), all_dice
    #         )


def load_cases():
    test_list = pd.read_csv("test_files_with_mask_class_info_sorted_by_pid.csv")

    test_list.head()

    patient_list = test_list.pid.unique()
    print(patient_list)
    return patient_list


def main():
    args = get_args()
    global base_dir
    base_dir = args.base_dir

    patient_list = load_cases()
    run_multiple_scenarios(patient_list, save_masks=args.save_masks)


def get_args():
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(prog="{0}".format(os.path.basename(sys.argv[0])))

    ap.add_argument(
        "--base-dir",
        type=str,
        required=False,
        default="data",
        help="Path to the directory containing the image and mask subdirectories.",
    )

    ap.add_argument(
        "--save-predictions",
        dest="save_masks",
        type=str,
        required=False,
        help="Path to a directory where predicted masks should be output; if not provided, masks will not be saved.",
    )

    return ap.parse_args()


if __name__ == "__main__":
    main()
