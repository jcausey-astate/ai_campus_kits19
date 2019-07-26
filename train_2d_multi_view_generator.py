#!/usr/bin/env python3
"""
Training script that supports K/T and KT/T models in any of the
three views {axial,coronal,saggital}.  Code is optimized for 
axial model training.

Run with --help to see command-line options.
"""
from __future__ import print_function

import os
from skimage.transform import resize, rescale
from skimage.io import imsave
import numpy as np
import pandas as pd
from keras.models import Model
from keras.utils import Sequence
from keras.layers import (
    Input,
    concatenate,
    Conv1D,
    MaxPooling1D,
    Conv2DTranspose,
    Lambda,
)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
import cv2
import sys

import unet
import random
from kits_volume_utils import *


# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))
K.set_image_data_format("channels_last")  # TF dimension ordering in this code

SIZE = 512
BATCH_SIZE = 12
SLICES_PER_SAMPLE = 1
SLICES_PER_MASK = 1
DEBUG = True


def preprocess(imgs):
    return np.expand_dims(imgs, -1)


class NumpyMaskedVolumeGenerator(Sequence):
    def __init__(
        self,
        patient_info_dataframe,
        base_dir="",
        img_dir="vol",
        mask_dir="seg",
        batch_size=BATCH_SIZE,
        seed=1,
        target_size=(512, 512),
        channels=1,
        layer_per_class=True,
        num_class=2,
        augment=True,
        slices_per_sample=SLICES_PER_SAMPLE,
        debug_output=DEBUG,
        return_ids=False,
        neg_pct=34,
        orientation="axial",
        id_col_name="pid",
        kt_t_mask_format=False,
    ):
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = True
        self.target_size = target_size
        self.shape = target_size + (channels,)
        if slices_per_sample is not None and slices_per_sample > 1:
            self.shape = (slices_per_sample,) + self.shape
        self.channels = channels
        self.last_index = -1
        self.base_dir = base_dir
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.layer_per_class = layer_per_class
        self.num_class = num_class
        self.augment = augment
        self.slices_per_sample = slices_per_sample
        self.first_batch = True
        self.debug = debug_output
        self.return_ids = return_ids
        self.neg_pct = neg_pct if float(neg_pct) <= 1 else (neg_pct / 100.0)
        self.orientation = orientation.lower()[0]
        self.id_col_name = id_col_name
        self.kt_t_mask_format = kt_t_mask_format
        if self.orientation not in ["a", "c", "s"]:
            raise RuntimeError(
                'Unknown orientation: Orientation must be "axial", "coronal", or "saggital".'
            )
        self.cache = {"sid": None, "vol": None, "seg": None}
        try:
            patient_info_dataframe = patient_info_dataframe.sort_values(
                by=[self.id_col_name]
            )
        except:
            raise RuntimeError(
                "Patient metadata must be supplied as a Pandas dataframe with id column"
                + " that matches the name passed int the parameter. "
                + " The supplied metadata failed to process."
            )
        self.pids = list(patient_info_dataframe[self.id_col_name].unique())
        self.sample_list = self.pids  # alias

        # Do this last...
        self.on_epoch_end()

    def __len__(self):
        """Guess at the number of batches per epoch from approx size of inputs."""
        # This version assumes 1 epoch gives us a batch of random slices from each
        # patient once.
        return len(self.sample_list)

    def __load_volumes(self, pid):
        vol, seg = None, None
        if self.cache is not None and "pid" in self.cache and self.cache["pid"] == pid:
            # Use cached volume
            vol = self.cache["vol"]
            seg = self.cache["seg"]
        else:
            # Cache a new volume:
            # print("Caching volume for PID {} base dir {} img dir {} mask_dir {} orientation {}".format(pid, self.base_dir, self.img_dir, self.mask_dir, self.orientation))
            vol, seg = load_3d_volume(
                pid,
                base_dir=self.base_dir,
                img_dir=self.img_dir,
                mask_dir=self.mask_dir,
                orientation=self.orientation,
            )
            self.cache = {"pid": pid, "vol": vol, "seg": seg}
        return vol, seg

    def __getitem__(self, index):
        """
        Generate one batch of data.
        To be efficient here, we will try to load all the necessary samples 
        from a single volume; if we can't, then we will group with the next (random)
        volume, etc.
        """
        # print("\n__getitem__ is being called with index {}".format(index))
        # The `index` corresponds to a desired PID, but we will rotate the list
        # so that we can use more than one if we need to.
        sub_list = np.roll(self.sample_list, -index)

        # Now start with the first and try to satisfy the batch size and neg percent
        # constraints.
        collected_samples = 0
        required_neg = int(round(self.batch_size * self.neg_pct))
        required_pos = self.batch_size - required_neg
        batch = []
        while (required_pos > 0 or required_neg > 0) and len(sub_list) > 0:
            pid, sub_list = sub_list[0], sub_list[1:]
            vol, seg = self.__load_volumes(pid)
            pos_slices = np.arange(vol.shape[0])
            neg_slices = np.copy(pos_slices)

            if seg is not None:
                all_slices = set(np.arange(seg.shape[0]))
                pos_slices = set(np.any(seg > 0, axis=(1, 2)).nonzero()[0])
                neg_slices = list(all_slices - pos_slices)
                pos_slices = np.asarray(list(pos_slices))
                neg_slices = np.asarray(neg_slices)
                np.random.shuffle(pos_slices)
                np.random.shuffle(neg_slices)

            n_pos, n_neg = len(pos_slices), len(neg_slices)

            if required_pos > n_pos:
                batch.extend([(pid, slc) for slc in pos_slices])
                required_pos -= n_pos
            elif required_pos > 0:
                batch.extend(
                    [
                        (pid, slc)
                        for slc in np.random.choice(
                            pos_slices, required_pos, replace=False
                        )
                    ]
                )
                required_pos = 0
            if required_neg > n_neg:
                batch.extend([(pid, slc) for slc in neg_slices])
                required_neg -= n_neg
            elif required_neg > 0:
                batch.extend(
                    [
                        (pid, slc)
                        for slc in np.random.choice(
                            neg_slices, required_neg, replace=False
                        )
                    ]
                )
                required_neg = 0

        # Generate data
        X, y = self.__data_generation(batch)

        return X, y

    def on_epoch_end(self):
        """Updates indices after each epoch"""

        if self.seed is not None:
            orig_state = np.random.get_state()
            np.random.seed(self.seed)

        self.indices = np.arange(len(self.sample_list))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

        if self.seed is not None:
            self.seed = np.random.randint(2 ** 32)
            np.random.set_state(orig_state)

    def __check_and_resize(self, img, is_mask=False):
        if img.shape != self.target_size:
            if not is_square(img.shape):
                img = make_square_img(img)
            img = resize_to_shape(img, self.target_size, is_mask=is_mask)
        return img

    def __simple_aug(self, img, mask):
        lr_roll = np.random.random()
        ud_roll = np.random.random()
        amount = int(round(self.target_size[0] * 0.15))
        tx = np.random.randint(amount * 2) - amount
        ty = np.random.randint(amount * 2) - amount
        translate_roll = np.random.random()
        rescale_roll = np.random.random()
        if lr_roll < 0.5:
            img = np.fliplr(img)
            if mask is not None:
                mask = np.fliplr(mask)
        # if ud_roll < 0.5:
        #     img = np.flipud(img)
        #     if mask is not None:
        #         mask = np.flipud(mask)
        if translate_roll < 0.5:
            img = img_translate(img, tx, ty)
            if mask is not None:
                mask = img_translate(mask, tx, ty)
        if rescale_roll < 0.5:
            scale = np.random.uniform(0.85, 1.1501)
            img = rescale_and_crop(img, scale, is_mask=False)
            if mask is None:
                mask = rescale_and_crop(mask, scale, is_mask=True)

        return img, mask

    def __data_generation(self, gen_list):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.shape))
        y = None
        if self.mask_dir is not None:
            y = np.empty((self.batch_size, *self.shape), dtype="uint8")
        if self.layer_per_class:
            y = np.empty(
                (self.batch_size, *((self.target_size) + (self.num_class,))),
                dtype="uint8",
            )

        assert len(gen_list) >= 1, "Data generation invoked on an empty gen_list!"

        # Generate data
        for idx, sample_info in enumerate(gen_list):
            # Each sample_info is a tuple of pid, slice_idx
            pid, slice_idx = sample_info
            vol, seg = self.__load_volumes(pid)

            img = vol[slice_idx]
            mask = None
            if seg is not None:
                mask = seg[slice_idx]

            img = self.__check_and_resize(img)

            if seg is not None:
                mask = self.__check_and_resize(mask, is_mask=True)

            img = np.reshape(img, img.shape + (1,))

            # Apply flips if requested:
            if self.augment:
                img, mask = self.__simple_aug(img, mask)

            # Store sample and mask:
            X[idx,] = img
            if mask is not None:
                mask = self.adjust_mask(mask)
                y[idx,] = mask

        if self.debug and self.first_batch:
            debug_dir = os.path.join(".", "logs", "debug", "first_batch")
            os.makedirs(debug_dir, exist_ok=True)
            for idx, sample_info in enumerate(gen_list):
                pid, slc = sample_info
                f_base_name = "{}_{}".format(pid, slc)
                np.save(
                    os.path.join(debug_dir, "{}_X.npy".format(f_base_name)), X[idx,]
                )
                if seg is not None:
                    np.save(
                        os.path.join(debug_dir, "{}_y.npy".format(f_base_name)), y[idx,]
                    )
            self.first_batch = False

        result = (X, y) if mask is not None else (X,)
        if self.return_ids:
            result = result + (gen_list,)
        return result

    def adjust_mask(self, mask):
        return adjust_mask(
            mask,
            flag_multi_class=self.layer_per_class,
            num_class=self.num_class,
            kt_t_mask_format=self.kt_t_mask_format,
        )


class NumpyMaskedAxialSliceGenerator(Sequence):
    """This version is simpler and faster for axial slices."""

    def __init__(
        self,
        patient_info_dataframe,
        base_dir="",
        img_dir="vol",
        mask_dir="seg",
        batch_size=BATCH_SIZE,
        seed=1,
        target_size=(512, 512),
        channels=1,
        layer_per_class=True,
        num_class=2,
        augment=True,
        debug_output=DEBUG,
        return_ids=False,
        neg_pct=34,
        id_col_name="pid",
        class_col_name="class",
        slc_col_name="slice",
        kt_t_mask_format=False,
    ):
        self.patient_info = patient_info_dataframe
        self.base_dir = base_dir
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.seed = seed
        self.shape = target_size + (channels,)
        self.target_size = target_size
        self.channels = channels
        self.layer_per_class = layer_per_class
        self.num_class = num_class
        self.augment = augment
        self.debug_output = debug_output
        self.return_ids = return_ids
        self.id_col_name = id_col_name
        self.class_col_name = class_col_name
        self.slc_col_name = slc_col_name
        self.kt_t_mask_format = kt_t_mask_format
        self.shuffle = mask_dir is not None
        self.neg_pct = neg_pct / 100.0 if neg_pct >= 1 else neg_pct

        try:
            patient_info_dataframe = patient_info_dataframe.sort_values(
                by=[self.id_col_name, self.class_col_name, self.slc_col_name]
            )
        except:
            raise RuntimeError(
                "Patient metadata must be supplied as a Pandas dataframe with id, class, "
                + "and slice index columns that match the names passed as parameters. "
                + " The supplied metadata failed to process."
            )
        self.pids = list(patient_info_dataframe[self.id_col_name].unique())
        self.sample_list = []
        self.__prepare_sample_list()

        # Do this last...
        self.on_epoch_end()

    def __preprocess_sample_list(self):
        orig_list = self.sample_list

    def __len__(self):
        """Guess at the number of batches per epoch from approx size of inputs."""
        return len(self.sample_list) // self.batch_size

    def __load_slice(self, fname):
        img, seg = None, None
        img, seg = load_numpy_img_and_mask(
            fname=fname,
            base_dir=self.base_dir,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            target_size=self.target_size,
            flag_multi_class=False,
            num_class=self.num_class,
            do_adjust=False,
        )
        if len(img.shape) > 2:
            img = np.squeeze(img)
        if len(seg.shape) > 2:
            seg = np.squeeze(seg)
        return img, seg

    def __prepare_sample_list(self):
        # Now start with the first and try to satisfy the batch size and neg percent
        # constraints.
        collected_samples = 0
        neg_pct = self.neg_pct
        pos_pct = 1.0 - neg_pct
        total_available = len(self.patient_info.index)
        pos_slices = self.patient_info[self.patient_info[self.class_col_name] == 1]
        neg_slices = self.patient_info[self.patient_info[self.class_col_name] == 0]

        available_pos = len(pos_slices.index)
        available_neg = total_available - available_pos

        if available_pos < available_neg:
            required_pos = available_pos
            neg_ratio = neg_pct / pos_pct
            required_neg = min(int(round(neg_ratio * required_pos)), available_neg)
        else:
            required_neg = available_neg
            pos_ratio = pos_pct / neg_pct
            required_pos = min(int(round(pos_ratio * required_neg)), available_pos)

        print(
            "Using {} positive (of {}) and {} negative (of {}) examples.".format(
                required_pos, available_pos, required_neg, available_neg
            )
        )

        sample_size = required_pos + required_neg

        selected_pos_slices = pos_slices.sample(required_pos, replace=False)
        selected_neg_slices = neg_slices.sample(required_neg, replace=False)
        selected_slices = pd.concat([selected_pos_slices, selected_neg_slices])

        self.sample_list = [
            (r[self.id_col_name], r[self.slc_col_name])
            for i, r in selected_slices.iterrows()
        ]

        np.random.shuffle(self.sample_list)

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        # The `index` corresponds to a desired PID, but we will rotate the list
        # so that we can use more than one if we need to.
        batch = self.sample_list[
            index
            * self.batch_size : min(
                len(self.sample_list), (index + 1) * self.batch_size
            )
        ]

        # Generate data
        X, y = self.__data_generation(batch)

        return X, y

    def on_epoch_end(self):
        """Updates indices after each epoch"""

        if self.seed is not None:
            orig_state = np.random.get_state()
            np.random.seed(self.seed)

        self.indices = np.arange(len(self.sample_list))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

        if self.seed is not None:
            self.seed = np.random.randint(2 ** 32)
            np.random.set_state(orig_state)

    def __check_and_resize(self, img, is_mask=False):
        if img.shape != self.target_size:
            if not is_square(img.shape):
                img = make_square_img(img)
            img = resize_to_shape(img, self.target_size, is_mask=is_mask)
        return img

    def __simple_aug(self, img, mask):
        lr_roll = np.random.random()
        ud_roll = np.random.random()
        amount = int(round(self.target_size[0] * 0.15))
        tx = np.random.randint(amount * 2) - amount
        ty = np.random.randint(amount * 2) - amount
        translate_roll = np.random.random()
        rescale_roll = np.random.random()
        if lr_roll < 0.5:
            img = np.fliplr(img)
            if mask is not None:
                mask = np.fliplr(mask)
        # if ud_roll < 0.5:
        #     img = np.flipud(img)
        #     if mask is not None:
        #         mask = np.flipud(mask)
        if translate_roll < 0.5:
            img = img_translate(img, tx, ty)
            if mask is not None:
                mask = img_translate(mask, tx, ty)
        if rescale_roll < 0.5:
            scale = np.random.uniform(0.85, 1.1501)
            img = rescale_and_crop(img, scale, is_mask=False)
            if mask is None:
                mask = rescale_and_crop(mask, scale, is_mask=True)

        return img, mask

    def __data_generation(self, gen_list):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.shape))
        y = None
        if self.mask_dir is not None:
            y = np.empty((self.batch_size, *self.shape), dtype="uint8")
        if self.layer_per_class:
            y = np.empty(
                (self.batch_size, *((self.target_size) + (self.num_class,))),
                dtype="uint8",
            )
        assert len(gen_list) >= 1, "Data generation invoked on an empty gen_list!"

        # Generate data
        for idx, sample_info in enumerate(gen_list):
            pid, slc = sample_info
            fname = "{}_{}.npy".format(pid, slc)
            img, mask = self.__load_slice(fname)

            img = self.__check_and_resize(img)

            if mask is not None:
                mask = self.__check_and_resize(mask, is_mask=True)

            # Apply flips if requested:
            if self.augment:
                img, mask = self.__simple_aug(img, mask)

            # Store sample and mask:
            img = np.reshape(img, img.shape + (1,))
            X[idx,] = img
            if mask is not None:
                mask = self.adjust_mask(mask)
                y[idx,] = mask

        result = (X, y) if mask is not None else (X,)
        if self.return_ids:
            result = result + (gen_list,)
        return result

    def adjust_mask(self, mask):
        return adjust_mask(
            mask,
            flag_multi_class=self.layer_per_class,
            num_class=self.num_class,
            kt_t_mask_format=self.kt_t_mask_format,
        )


def main():
    args = get_args()
    model = unet.get_unet()

    if args.kt_t_mask_format:
        print("\nUSING KT/T MASK FORMAT\n")

    if args.existing_weights is not None and os.path.isfile(args.existing_weights):
        print(
            "\nResuming training from {} - Epoch {} to {}".format(
                args.existing_weights, args.resume_from_epoch, args.n_epochs
            )
        )
        model.load_weights(args.existing_weights, by_name=True)

    train_meta = pd.read_csv(args.train_info_file)
    # Pull off 20% for test set
    train_meta = train_meta.sort_values(by="pid")
    pids = np.asarray(train_meta["pid"].unique())
    np.random.shuffle(pids)
    train_set_size = int(round(len(pids) * 0.8))
    train_pids, test_pids = pids[:train_set_size], pids[train_set_size:]
    train_train_meta = train_meta[train_meta["pid"].isin(train_pids)]
    train_test_meta = train_meta[train_meta["pid"].isin(test_pids)]

    checkpoint_name = (
        "unet_KiTS{}__{}".format(
            "" if not args.kt_t_mask_format else "~KT-T", args.orientation
        )
        + "_e{epoch:02d}-l{dice_coef_loss:.2f}-vl{val_dice_coef_loss:.2f}.h5"
    )
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir="./logs", histogram_freq=0, write_graph=True, write_images=False
        )
    ]
    if not args.save_last_only:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                os.path.join("./checkpoints", checkpoint_name),
                verbose=0,
                save_weights_only=True,
            )
        )

    print("Training data head:\n{}".format(train_train_meta.head()))

    print("\nVal data head:\n{}".format(train_test_meta.head()))

    if args.orientation.lower()[0] == "a":
        print("\nAXIAL orientation using slice-oriented generator for speed.\n")
        # Axial model uses the faster slice-oriented generator:
        train_gen = NumpyMaskedAxialSliceGenerator(
            train_train_meta,
            base_dir=args.base_dir,
            img_dir=args.image_dir,
            mask_dir=args.mask_dir,
            batch_size=args.batch_size,
            id_col_name="pid",
            slc_col_name="slice",
            class_col_name="has_either",
            kt_t_mask_format=args.kt_t_mask_format,
        )

        test_gen = NumpyMaskedAxialSliceGenerator(
            train_test_meta,
            base_dir=args.base_dir,
            img_dir=args.image_dir,
            mask_dir=args.mask_dir,
            batch_size=args.batch_size,
            id_col_name="pid",
            slc_col_name="slice",
            class_col_name="has_either",
            kt_t_mask_format=args.kt_t_mask_format,
        )

    else:
        train_gen = NumpyMaskedVolumeGenerator(
            train_train_meta,
            base_dir=args.base_dir,
            img_dir=args.image_dir,
            mask_dir=args.mask_dir,
            batch_size=args.batch_size,
            slices_per_sample=1,
            orientation=args.orientation,
            id_col_name="pid",
            kt_t_mask_format=args.kt_t_mask_format,
        )

        test_gen = NumpyMaskedVolumeGenerator(
            train_test_meta,
            base_dir=args.base_dir,
            img_dir=args.image_dir,
            mask_dir=args.mask_dir,
            batch_size=args.batch_size,
            slices_per_sample=1,
            orientation=args.orientation,
            id_col_name="pid",
            kt_t_mask_format=args.kt_t_mask_format,
        )

    model.fit_generator(
        train_gen,
        steps_per_epoch=len(train_gen) if args.steps is None else args.steps,
        nb_epoch=args.n_epochs,
        validation_data=test_gen,
        validation_steps=100,
        callbacks=callbacks,
        initial_epoch=args.resume_from_epoch,
    )
    if args.save_last_only:
        model.save_weights(os.path.join(".", "weights.h5"))


def get_args():
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(prog="{0}".format(os.path.basename(sys.argv[0])))
    ap.add_argument(
        "--resume-training",
        dest="existing_weights",
        type=str,
        required=False,
        help="Provide path to existing weights to continue training.",
    )
    ap.add_argument(
        "--epochs",
        dest="n_epochs",
        type=int,
        required=False,
        default=8,
        help="Number of epochs to train.",
    )
    ap.add_argument(
        "--resume-from-epoch",
        type=int,
        required=False,
        default=0,
        help="Number of the last epoch that was trained, for resuming training.",
    )
    ap.add_argument(
        "--steps",
        dest="steps",
        type=int,
        required=False,
        help="Number of steps per epoch to train.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=BATCH_SIZE,
        help="Batch size for training.",
    )
    ap.add_argument(
        "--training-samples",
        dest="train_info_file",
        type=str,
        required=False,
        default="train_files_ordered_by_patient.csv",
        help="Text file containing list of samples for training.",
    )
    ap.add_argument(
        "--view",
        dest="orientation",
        type=str,
        required=False,
        default="axial",
        choices=["axial", "coronal", "saggital", "a", "c", "s"],
        help="View to use for training (axial, coronal, saggital).",
    )
    ap.add_argument(
        "--save-last-only",
        action="store_true",
        help="Save final weights only; do not checkpoint.",
    )
    ap.add_argument(
        "--kt-t-mask-format",
        action="store_true",
        help="Use KT/T mask format instead of K/T masks.",
    )
    ap.add_argument(
        "--base-dir",
        type=str,
        required=False,
        default="data",
        help="Path to the directory containing the image and mask subdirectories.",
    )
    ap.add_argument(
        "--image-dir",
        type=str,
        required=False,
        default="vol",
        help="Name of the image subdirectory, relative to the --base-dir.",
    )
    ap.add_argument(
        "--mask-dir",
        type=str,
        required=False,
        default="seg",
        help="Name of the mask subdirectory, relative to the --base-dir.",
    )

    return ap.parse_args()


if __name__ == "__main__":
    main()
