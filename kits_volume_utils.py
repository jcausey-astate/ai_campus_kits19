"""
Library to provide many functions for loading, manipulating, 
rotating, cropping, etc. on volumetric data in Numpy format.
Also functions for loading from NiFTi.
"""
from __future__ import print_function
import numpy as np
import copy
import os, sys, shutil, glob
from skimage.transform import resize, rescale
import imageio

Background = [0, 0, 0]
Kidney = [255, 0, 0]
Tumor = [0, 0, 255]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Background, Kidney, Tumor, Unlabelled])

SLICES_PER_SAMPLE = 1
SLICES_PER_MASK = 1

DEBUG = True


def load_numpy_img_and_mask(
    fname,
    base_dir="",
    img_dir="",
    mask_dir=None,
    target_size=(512, 512),
    augs=None,
    flag_multi_class=False,
    num_class=2,
    do_adjust=True,
):
    """ Loads the numpy image and corresponding mask (unless mask_dir is None)"""
    img, seg = load_raw_img_seg(fname, base_dir, img_dir, mask_dir)

    # Resize the image if necessary
    if img.shape != target_size:
        if is_square(target_size):
            img = make_square_img(img)
            if seg is not None:
                seg = make_square_img(seg)
        img = resize_to_shape(img, target_size)
        if seg is not None:
            seg = resize_to_shape(seg, target_size, is_mask=True)

    # Apply augmentations if requested:
    if augs is not None:
        seg_dtype = seg.dtype
        img, seg = augs(image=img, segmentation_maps=seg.astype(int))
        seg = seg.astype(seg_dtype)

    # Adjust the mask and image if necessary:
    if seg is not None and do_adjust:
        seg = adjust_mask(seg, flag_multi_class=flag_multi_class, num_class=num_class)
        if len(seg.shape) < 3:
            seg = np.reshape(seg, seg.shape + (1,))
    if len(img.shape) < 3:
        img = np.reshape(img, img.shape + (1,))

    return img, seg


def load_numpy_img_and_mask_stack(
    fname,
    img_stack_size=SLICES_PER_SAMPLE,
    seg_stack_size=SLICES_PER_MASK,
    base_dir="",
    img_dir="",
    mask_dir=None,
    target_size=(512, 512),
    augs=None,
    flag_multi_class=False,
    num_class=2,
    is_timeseries=False,
):

    # The current filename is always the "middle".
    assert img_stack_size % 2 == 1, "img_stack_size must be an odd number."
    assert (
        seg_stack_size == 1 or seg_stack_size == img_stack_size
    ), "seg_stack_size must be 1 or same as img_stack"
    seg_target_size = target_size
    if flag_multi_class:
        seg_target_size = seg_target_size + (num_class,)
    img_stack = np.zeros(target_size + (img_stack_size,))
    seg_stack = np.zeros(seg_target_size + (seg_stack_size,))
    img_stack_mid = img_stack_size // 2
    seg_stack_mid = img_stack_mid if seg_stack_size > 1 else 0

    # Names are like 123_45.npy first part is PID, second is idx
    cur_pid, cur_idx = (
        int(v) for v in os.path.splitext(os.path.basename(fname))[0].split("_")[:2]
    )

    min_idx = cur_idx - (img_stack_size // 2)
    max_idx = cur_idx + int(np.ceil(img_stack_size / 2.0))

    min_seg_idx = cur_idx if seg_stack_size == 1 else min_idx
    max_seg_idx = cur_idx + 1 if seg_stack_size == 1 else max_idx

    # NOTE: It is important here that the same augmentation is perfomed
    #       on every image and every mask in the stack.  To ensure this,
    #       we need to copy and re-apply the random state after each
    #       operation.
    img_augs = augs
    if img_augs is not None:
        img_augs = img_augs.localize_random_state()
        img_augs.random_state.seed(np.random.randint(999999))
        seed_state = mask_augs.copy_random_state(img_augs)

    # Set the middle slice first.  This should never fail (or if it does, it should be a crasher).
    if img_augs is not None:
        img_augs = img.augs.copy_random_state(seed_state)
    img, seg = load_numpy_img_and_mask(
        fname,
        base_dir=base_dir,
        img_dir=img_dir,
        mask_dir=mask_dir,
        target_size=target_size,
        augs=img_augs,
        flag_multi_class=flag_multi_class,
        num_class=num_class,
    )
    img_stack[..., img_stack_mid] = img[..., 0]

    # print("seg_shape: {} seg_stack shape: {}".format(seg.shape, seg_stack.shape))

    seg_stack[..., seg_stack_mid] = (
        seg[..., 0] if len(seg.shape) > (len(seg_stack.shape) + 1) else seg
    )

    # Set the rest if possible -- if slices don't exist (edges), keep zeros.
    for stack_idx, slice_idx in enumerate(range(min_idx, max_idx)):
        if stack_idx == img_stack_mid:  # middle slice is already set.
            continue
        slice_name = "{}_{}.npy".format(cur_pid, slice_idx)
        if not os.path.isfile(os.path.join(base_dir, img_dir, slice_name)):
            img = img_stack[:, :, stack_idx]
            seg = seg_stack[:, :, stack_idx if seg_stack_size > 1 else 0]
        else:
            if img_augs is not None:
                img_augs = img.augs.copy_random_state(seed_state)
            img, seg = load_numpy_img_and_mask(
                slice_name,
                base_dir=base_dir,
                img_dir=img_dir,
                mask_dir=mask_dir,
                target_size=target_size,
                augs=img_augs,
                flag_multi_class=flag_multi_class,
                num_class=num_class,
            )
        img_stack[..., stack_idx] = img[..., 0]
        if seg_stack_size > 1:
            seg_stack[..., stack_idx] = seg[..., 0]

    if seg_stack_size == 1:  # if there is only one seg, unstack it.
        seg_stack = seg_stack[..., 0]

    if is_timeseries:
        img_stack = np.moveaxis(img_stack, -1, 0)
        if seg_stack_size > 1:
            seg_stack = np.moveaxis(seg_stack, -1, 0)
    # print(
    #     "Returning {}stack of shape {}".format(
    #         "timeseries " if is_timeseries else "", img_stack.shape
    #     )
    # )

    return img_stack, seg_stack


def standardize_HU(vol, HU_max=500, HU_min=-500):
    vol[vol > HU_max] = HU_max
    vol[vol < HU_min] = HU_min

    conversion_factor = 1.0 / (HU_max - HU_min)
    conversion_intercept = 0.5
    vol = vol * conversion_factor + conversion_intercept

    assert np.amax(vol) <= 1, "Max above one after normalization."
    assert np.amin(vol) >= 0, "Min below zero after normalization."

    return vol


def load_nifti_volume(pid, base_dir, is_mask=False, return_affine=False):
    import nibabel

    # print('Loading {}.'.format('case' if not is_mask else 'segmentation'))
    p_data_dir = "case_{:05d}".format(int(pid))
    p_data_file_name = "imaging.nii.gz" if not is_mask else "segmentation.nii.gz"
    path = os.path.join(base_dir, p_data_dir, p_data_file_name)
    volume = nibabel.load(path)
    affine = volume.affine
    volume = volume.get_fdata()

    if not is_mask:
        # Next, standardize Hounsfield units.
        # print('Standardizing Hounsfield Units.')
        volume = standardize_HU(volume)

    return volume if not return_affine else (volume, affine)


def load_numpy_img_and_mask_stack_axis(
    sample_name,
    slice_index=None,
    axis="axial",
    img_stack_size=SLICES_PER_SAMPLE,
    seg_stack_size=SLICES_PER_MASK,
    base_dir="",
    img_dir="",
    mask_dir=None,
    target_size=(512, 512),
    augs=None,
    flag_multi_class=False,
    num_class=2,
    is_timeseries=False,
    do_cache=False,
    cache=None,
):
    axis_values = ["axial", "coronal", "saggital"]
    # If we can hand off to the simpler "just load the slices involved"
    # method, do that.
    if axis[:1].lower() == "a":
        if not (sample_name[:-4] in [".npy", ".npz"] or sample_name[:-3] == ".np"):
            sample_name = "{}_{}.npy".format(sample_name, slice_index)

        img, seg = load_numpy_img_and_mask_stack(
            sample_name,
            img_stack_size=img_stack_size,
            seg_stack_size=seg_stack_size,
            base_dir=base_dir,
            img_dir=img_dir,
            mask_dir=mask_dir,
            target_size=target_size,
            augs=augs,
            flag_multi_class=flag_multi_class,
            num_class=num_class,
            is_timeseries=is_timeseries,
        )
        return (img, seg) if not do_cache else (img, seg, None)

    # Here, we must load up the entire image and seg volumes associated
    # with this sample.
    sample_name = os.path.splitext(os.path.basename(sample_name))[0]
    name_parts = sample_name.split("_")
    sample_name = parts[0]
    if slice_index is None:
        try:
            slice_index = parts[1]
        except:
            raise RuntimeError(
                "Slice index must be specified with axis = "
                + "'{}' and no slice embedded in sample name.".format(axis)
            )

    # Load the volume if we don't have it in cache already
    if (
        cache is None
        or type(cache) != dict
        or "sample_name" not in cache
        or cache["sample_name"] != sample_name
    ):
        img_vol, seg_vol = load_volumes_for_sample(
            sample_name,
            base_dir=base_dir,
            img_dir=img_dir,
            mask_dir=mask_dir,
            orientation=None,  # Don't re-orient here - cache is non-oriented.
        )
        cache = {"sample_name": sample_name, "img_vol": img_vol, "seg_vol": seg_vol}
    else:
        assert sample_name == cache["sample_name"], "Sample name must match cache here."
        img_vol = cache["img_vol"]
        seg_vol = cache["seg_vol"]

    # Now reorient the volume to the correct view:
    img_vol = reorient_volume(img_vol, axis)
    seg_vol = reorient_volume(seg_vol, axis)

    # The current sample/slice info is always the "middle".
    assert img_stack_size % 2 == 1, "img_stack_size must be an odd number."
    assert (
        seg_stack_size == 1 or seg_stack_size == img_stack_size
    ), "seg_stack_size must be 1 or same as img_stack"
    seg_target_size = target_size
    if flag_multi_class:
        seg_target_size = seg_target_size + (num_class,)
    img_stack = np.zeros(target_size + (img_stack_size,))
    seg_stack = np.zeros(seg_target_size + (seg_stack_size,))
    img_stack_mid = img_stack_size // 2
    seg_stack_mid = img_stack_mid if seg_stack_size > 1 else 0

    cur_pid = sample_name
    cur_idx = int(slice_no)

    min_idx = cur_idx - (img_stack_size // 2)
    max_idx = cur_idx + int(np.ceil(img_stack_size / 2.0))

    min_seg_idx = cur_idx if seg_stack_size == 1 else min_idx
    max_seg_idx = cur_idx + 1 if seg_stack_size == 1 else max_idx

    # NOTE: It is important here that the same augmentation is perfomed
    #       on every image and every mask in the stack.  To ensure this,
    #       we need to copy and re-apply the random state after each
    #       operation.
    img_augs = augs
    if img_augs is not None:
        img_augs = img_augs.localize_random_state()
        img_augs.random_state.seed(np.random.randint(999999))
        seed_state = img_augs.copy_random_state(img_augs)

    # Fill the output stack with images from the volume.
    for stack_idx, slice_idx in enumerate(range(min_idx, max_idx)):
        # Just keep zeros for outside-the-volume slices
        if slice_idx < 0 or slice_idx >= img_vol.shape[-1]:
            continue

        # Everything else gets sized and placed:
        img = resize_to_shape(img_vol[..., stack_idx], target_size)
        seg = None

        if (seg_vol is not None) and (stack_idx == img_stack_mid or seg_stack_size > 1):
            seg = resize_to_shape(seg_vol[..., stack_idx], target_size, is_mask=True)

        # Augment if required
        if img_augs is not None:
            # Use the same transform for every image in the stack!!!
            img_augs = img_augs.copy_random_state(seed_state)
            if seg is not None:
                img, seg = img_augs(image=img, segmentation_maps=seg)
            else:
                img = img_augs(image=img)

        img_stack[..., stack_idx] = img
        # Place the seg only if we have one:
        if seg is not None:
            seg_stack_idx = 0 if seg_stack_size == 1 else stack_idx
            seg_stack[..., seg_stack_idx] = seg

    # if there is only one seg, unstack it.
    if seg_vol is not None and seg_stack_size == 1:
        seg_stack = seg_stack[..., 0]

    if is_timeseries:
        img_stack = np.moveaxis(img_stack, -1, 0)
        if seg_vol is not None and seg_stack_size > 1:
            seg_stack = np.moveaxis(seg_stack, -1, 0)

    return img_stack, seg_stack if not do_cache else img_stack, seg_stack, cache


def load_3d_volume(
    sample_name,
    base_dir="",
    img_dir="",
    mask_dir=None,
    orientation=None,
    use_nifti=False,
):
    """
    Loads the corresponding image and (if mask_dir is not None) seg volumes
    for the sample, assuming they are available as single 3-d volumetric
    Numpy files.  Returns them in the requested orientation such that the 
    z-dimension is "first", i.e.  (z,y,x) ordering.
    This works given a volumetric name such as PID.npy or a 2-d PID_SLC.npy
    name.
    """
    sample_name = sample_name_from_filename(sample_name)

    img_vol = None
    seg_vol = None

    if not use_nifti:
        if img_dir is not None:
            img_vol_path = os.path.join(base_dir, img_dir, "{}.npy".format(sample_name))
            img_vol = np.load(img_vol_path)
            img_vol = np.moveaxis(img_vol, [0, 1, 2], [1, 2, 0])

        if mask_dir is not None:
            seg_vol_path = os.path.join(
                base_dir, mask_dir, "{}.npy".format(sample_name)
            )
            seg_vol = np.load(seg_vol_path)
            seg_vol = np.moveaxis(seg_vol, [0, 1, 2], [1, 2, 0])
    else:
        img_vol = load_nifti_volume(
            sample_name, base_dir, is_mask=False, return_affine=False
        )
        seg_vol = load_nifti_volume(
            sample_name, base_dir, is_mask=True, return_affine=False
        )

    if orientation is not None:
        # Now reorient the volume to the correct view:
        if img_vol is not None:
            img_vol = reorient_volume(img_vol, orientation)
        if seg_vol is not None:
            seg_vol = reorient_volume(seg_vol, orientation)

    return img_vol, seg_vol


def load_volumes_for_sample(
    sample_name, base_dir="", img_dir="", mask_dir=None, orientation=None
):
    """
    Loads the corresponding image and (if mask_dir is not None) seg volumes
    for the sample.  Returns them in the requested orientation such that the 
    z-dimension is "first", i.e.  (z,y,x) ordering.
    """
    # Here, we must load up the entire image and seg volumes associated
    # with this sample.
    sample_name = sample_name_from_filename(sample_name)

    glob_dir = img_dir if img_dir is not None else mask_dir

    # Get the list of files involved:
    vol_files = glob.glob(
        os.path.join(base_dir, glob_dir, "{}*.npy".format(sample_name))
    )
    vol_files = sorted(
        vol_files,
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[1]),
    )
    # The Z-axis size of the volume is determined by the number of files.
    z_shape = len(vol_files)

    # Load the first img and mask:
    img, seg = load_raw_img_seg(vol_files[0], base_dir, img_dir, mask_dir)
    y_shape, x_shape = img.shape if img is not None else seg.shape
    img_vol = None
    if img is not None:
        img_vol = np.zeros((z_shape, y_shape, x_shape), dtype=img.dtype)
    seg_vol = None
    if seg is not None:
        seg_vol = np.zeros((z_shape, y_shape, x_shape), dtype=seg.dtype)

    if img is not None:
        img_vol[0, ...] = img
    if seg is not None:
        seg_vol[0, ...] = seg
    # Now load all the rest of the slices
    for i in range(1, z_shape):
        img, seg = load_raw_img_seg(vol_files[i], base_dir, img_dir, mask_dir)
        if img is not None:
            img_vol[i, ...] = img
        if seg is not None:
            seg_vol[i, ...] = seg
    if orientation is not None:
        # Now reorient the volume to the correct view:
        if img_vol is not None:
            img_vol = reorient_volume(img_vol, orientation)
        if seg_vol is not None:
            seg_vol = reorient_volume(seg_vol, orientation)

    return img_vol, seg_vol


def adjust_mask(mask, flag_multi_class=False, num_class=2, kt_t_mask_format=False):
    """
    Use `kt_t_mask_format` if you want the first layer to contain Kidney+Tumor and
    the second layer to contain tumor segmentation only.  This implies `flag_multi_class`.
    """
    if kt_t_mask_format:
        flag_multi_class = True
    if flag_multi_class:
        # The masks are 512x512 (2d) and consist of 0,1,2 values at pixel locations.
        new_mask = np.zeros(mask.shape + (num_class,))
        # classes in KiTS mask are 1=kidney, 2=tumor 0 is background and we will ignore it.
        for i in range(num_class):
            new_mask[mask == (i + 1), i] = 1
        if kt_t_mask_format:
            new_mask[mask == 2, 0] = 1
        mask = new_mask
    else:
        # The masks are 512x512 (2d) and consist of 0,1,2 values at pixel locations.
        new_mask = np.zeros(mask.shape)
        # classes in KiTS mask are 1=kidney, 2=tumor 0 is background and we will ignore it.
        # If we don't want multi-class output, just combine kidney and tumor to create a "kidney+tumor" class.
        new_mask[mask > 0] = 1
        mask = new_mask
    return mask


def is_square(shape):
    return (shape[0] == shape[1]) and (len(shape) == 2)


def make_square_img(img, pad_value=0):
    """
    Makes image square, preserving the longest dimension.
    """
    if is_square(img.shape):
        return img
    maxdim = max(img.shape)
    pad_shape = tuple(
        [
            [np.ceil((maxdim - d) / 2.0).astype(int), (maxdim - d) // 2]
            for d in img.shape
        ]
    )
    return np.pad(img, pad_shape, "constant", constant_values=pad_value)


def make_square_volume(vol, pad_value=0):
    """
    Makes every image in the 'stack' dimenstion (z-dimension) square.
    Assumes (z,y,x) dim ordering.
    """
    if is_square(vol.shape):
        output_vol = vol
    else:
        maxdim = max(img.shape[1:])
        output_vol = np.zeros((img.shape[0],) + (maxdim, maxdim), dtype=vol.dtype)
        for idx, slc in enumerate(vol):
            output_vol[idx, ...] = make_square_img(slc, pad_value)
    return output_vol


def crop_square_img(img, cropped_shape):
    """
    This will undo the action of make_square_img, cropping evenly from
    the edges and leaving the center portion.
    """
    if img.shape == cropped_shape:
        return img

    current_shape = np.asarray(img.shape)
    cropped_shape = np.asarray(cropped_shape)
    delta = (current_shape - cropped_shape) / 2.0
    y_start = int(np.ceil(delta[0]))
    y_len = y_start + cropped_shape[0]
    x_start = int(np.ceil(delta[1]))
    x_len = x_start + cropped_shape[1]
    # print("y_start {} y_len {} x_start {} x_len {}".format(y_start,y_len,x_start,x_len))
    return img[y_start:y_len, x_start:x_len]


def crop_square_volume(vol, cropped_shape):
    """
    This will undo the action of make_square_volume, cropping evenly from
    the edges and leaving the center portion on a per-slice basis.  The
    `cropped_shape` should be a 2-d shape tuple.
    """
    if vol.shape[1:3] == cropped_shape:
        output_vol = vol
    else:
        output_vol = np.zeros((vol.shape[0],) + cropped_shape, dtype=vol.dtype)
        for idx, slc in enumerate(vol):
            output_vol[idx, ...] = crop_square_img(vol[idx, ...], cropped_shape)
    return output_vol


def resize_to_shape(img_to_resize, target_shape, is_mask=False):
    order = 3 if not is_mask else 0
    if not hasattr(target_shape, "__len__"):
        target_shape = (target_shape, target_shape)
    assert (
        len(target_shape) == 2
    ), "`target_shape` for resize_to_shape must be 2-D tuple or int."
    if img_to_resize.shape == target_shape:
        img = img_to_resize
    else:
        img_dtype = img_to_resize.dtype
        if is_square(target_shape):
            img_to_resize = make_square_img(img_to_resize)
        img = resize(
            img_to_resize.astype(float), target_shape, order=order, preserve_range=True
        )
        img = img.astype(img_dtype) if not is_mask else np.round(img).astype(img_dtype)
    return img


def resize_volume_slices(vol_to_resize, target_shape, is_mask=False):
    if not hasattr(target_shape, "__len__"):
        target_shape = (target_shape, target_shape)
    elif len(target_shape) > 2:
        target_shape = target_shape[1:3]
    if vol_to_resize[1:3] == target_shape:
        result_vol = vol_to_resize
    else:
        img_dtype = vol_to_resize.dtype
        result_vol = np.zeros((vol_to_resize.shape[0],) + target_shape, dtype=img_dtype)
        for idx, slc in enumerate(vol_to_resize):
            result_vol[idx, ...] = resize_to_shape(slc, target_shape, is_mask)
    return result_vol


def restore_volume_to_shape(vol_to_resize, orig_shape, is_mask=False):
    """Direct 'undo' of the order of operations from resize_to_shape."""
    if not hasattr(orig_shape, "__len__"):
        orig_shape = (orig_shape, orig_shape)
    elif len(orig_shape) > 2:
        orig_shape = orig_shape[1:3]
    if vol_to_resize.shape[1:3] == orig_shape:
        result_vol = vol_to_resize
    else:
        img_dtype = vol_to_resize.dtype
        # print(
        #     "Creating destination volume of shape {}".format(
        #         (vol_to_resize.shape[0],) + orig_shape
        #     )
        # )
        result_vol = np.zeros((vol_to_resize.shape[0],) + orig_shape, dtype=img_dtype)
        for idx, slc in enumerate(vol_to_resize):
            result_vol[idx, ...] = restore_to_shape(slc, orig_shape, is_mask)
    return result_vol


def restore_to_shape(img_to_resize, orig_shape, is_mask=False):
    """
    Restore a previously resized image to its original shape.  This is a direct
    inverse of operations from `resize_to_shape`.
    """
    order = 3 if not is_mask else 0
    if not hasattr(orig_shape, "__len__"):
        orig_shape = (orig_shape, orig_shape)
    assert len(orig_shape) == 2, "Image `orig_shape` must be a 2D shape tuple."

    if img_to_resize.shape == orig_shape:
        img = img_to_resize
    else:
        img_dtype = img_to_resize.dtype
        resize_inverse_shape = orig_shape
        current_shape = img_to_resize.shape
        if not is_square(orig_shape) and is_square(current_shape):
            # If the original wasn't square but the current one is, we need to
            # first scale it as a square, then later crop.
            resize_inverse_shape = (max(orig_shape), max(orig_shape))
        img = resize(
            img_to_resize.astype(float),
            resize_inverse_shape,
            order=order,
            preserve_range=True,
        )
        if not is_square(orig_shape) and is_square(current_shape):
            # If the original wasn't square but the current one is, we have resized,
            # but still need to crop.
            img = crop_square_img(img, orig_shape)
        img = img.astype(img_dtype) if not is_mask else np.round(img).astype(img_dtype)
        assert img.shape == orig_shape, "Restore to shape failed {} != {}.".format(
            img.shape, orig_shape
        )
    return img


def load_raw_img_seg(fname, base_dir="", img_dir="", mask_dir=None):
    img = None
    if img_dir is not None:
        img = np.load(os.path.join(base_dir, img_dir, fname))
    seg = None
    if mask_dir is not None:
        seg = np.load(os.path.join(base_dir, mask_dir, fname))
    return img, seg


def reorient_volume(vol, orientation, batch_order=True):
    assert orientation[0].lower() in [
        "a",
        "s",
        "c",
    ], 'Orientation must be one of ["axial", "coronal", "saggital"]'
    orientation = orientation[0].lower()

    if not batch_order:  # [y, x, z] ordering
        if orientation == "c":
            vol = np.moveaxis(vol, [0, 1, 2], [2, 1, 0])
        elif orientation == "s":
            vol = np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
    else:  # [z, y, x] ordering
        if orientation == "c":
            vol = np.moveaxis(vol, [0, 1, 2], [1, 0, 2])
        elif orientation == "s":
            vol = np.moveaxis(vol, [0, 1, 2], [1, 2, 0])

    return vol


def reorient_to_axial(vol, current_orientation, batch_order=True):
    assert current_orientation[0].lower() in [
        "a",
        "s",
        "c",
    ], 'Orientation must be one of ["axial", "coronal", "saggital"]'
    orientation = current_orientation[0].lower()

    if batch_order:
        if orientation == "c":
            vol = np.moveaxis(vol, [0, 1, 2], [1, 0, 2])
        elif orientation == "s":
            vol = np.moveaxis(vol, [0, 1, 2], [2, 0, 1])
    else:
        if orientation == "c":
            vol = np.moveaxis(vol, [0, 1, 2], [2, 1, 0])
        elif orientation == "s":
            vol = np.moveaxis(vol, [0, 1, 2], [2, 1, 0])

    return vol


def sample_name_from_filename(fname):
    try:
        sample_name = os.path.splitext(os.path.basename(fname))[0].split("_")[0]
    except:
        sample_name = fname
    return sample_name


def img_translate(img, tx, ty, channels_first=False):
    """
    Translates an image `img` by tx and ty units, padding with zeros.
    This method based on Daniel's method at https://stackoverflow.com/a/27087513
    Image is assumed to be "2d-like" with any number of channels.  Channels last.
    Use `channels_first` if channels are the first dimension.
    """
    neg_or_none = lambda s: s if s < 0 else None
    non_neg = lambda s: max(0, s)
    translated_img = np.zeros_like(img)
    if not channels_first:
        translated_img[
            non_neg(ty) : neg_or_none(ty), non_neg(tx) : neg_or_none(tx), ...
        ] = img[non_neg(-ty) : neg_or_none(-ty), non_neg(-tx) : neg_or_none(-tx), ...]
    else:
        translated_img[
            ..., non_neg(ty) : neg_or_none(ty), non_neg(tx) : neg_or_none(tx)
        ] = img[..., non_neg(-ty) : neg_or_none(-ty), non_neg(-tx) : neg_or_none(-tx)]
    return translated_img


def rescale_and_crop(img, scale, is_mask=False):
    """
    Rescales an image and crops/pads so that the result is the same pixel 
    dimensions as the original.
    
    @param      img      The image
    @param      scale    The scale
    @param      is_mask  Indicates if the image is a mask
    
    @return     Rescaled and cropped image.
    """
    img_rescaled = rescale(
        img,
        scale,
        order=1 if not is_mask else 0,
        mode="constant",
        cval=0,
        clip=True,
        preserve_range=True,
        anti_aliasing=True if not is_mask else False,
    )
    img_out = np.zeros_like(img)

    d_shape = np.asarray(img.shape) - np.asarray(img_rescaled.shape)
    start = d_shape // 2
    orig_shape = img.shape
    new_shape = img_rescaled.shape
    if scale < 1.0:
        img_out[
            start[0] : start[0] + new_shape[0], start[1] : start[1] + new_shape[1]
        ] = img_rescaled
    elif scale >= 1.0:
        start = [-s for s in start]
        img_out = img_rescaled[
            start[0] : start[0] + orig_shape[0], start[1] : start[1] + orig_shape[1]
        ]
    return img_out
