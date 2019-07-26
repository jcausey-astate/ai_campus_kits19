"""
Provides functions for post processing a 3D segmentation mask.
Assumes binary masks, but contains a driver for multi-label masks.
"""
import sys
import numpy as np
from skimage.morphology import remove_small_objects, convex_hull_object
from skimage.measure import label


def fill_gaps(vol, axis=0):
    """
    Fills gaps by iterating along `axis`, filling any 
    space where the "slice" above and below the current
    slice are both 1's.

    @param      vol   The volume
    @param      axis  The axis to iterate along.

    @return     The volume with gaps filled.
    """
    filled_vol = np.copy(vol)
    del vol
    if axis != 0:
        filled_vol = np.moveaxis(filled_vol, axis, 0)

    for idx in range(1, filled_vol.shape[0] - 1):
        filled_vol[idx, ...] = np.logical_or(
            filled_vol[idx, ...],
            np.logical_and(filled_vol[idx - 1, ...], filled_vol[idx + 1, ...]),
        ).astype(filled_vol.dtype)

    if axis != 0:
        filled_vol = np.moveaxis(filled_vol, 0, axis)

    return filled_vol


def fill_large_gaps(vol, depth=2, axis=0):
    """
    Fills larger gaps by iterating along `axis`, filling any 
    space where the "slice" above and below the current
    slice are both 1's, where the distance spanned by the 
    "gap" is determined by `depth`.

    @remark     This algoritm works by filling increasingly larger
                gaps, from 1 to `depth`.  That way you will not 
                encounter a fail to fill if a negative voxels happens
                to occur in a secondary gap where the "edges" of the
                large gap fill are positioned; e.g. with a depth of 3:
                idx:  0123456
                mask: 1011011
                The small gap at index 1 would fail to fill if we only
                consider sliding a window with span=3.  We must first 
                fill it with a smaller span; then the algorithm works.

    @param      vol   The volume
    @param      axis  The axis to iterate along.
    @param      depth The number of missing voxels that can be spanned.

    @return     The volume with gaps filled.
    """
    if axis is not None and axis != "all":
        vol = _fill_large_gaps(vol, depth, axis)
    else:
        for axis in range(len(vol.shape)):
            vol = _fill_large_gaps(vol, depth=depth, axis=axis)
    return vol


def _fill_large_gaps(vol, depth=2, axis=0):
    filled_vol = np.copy(vol)
    del vol
    if axis != 0:
        filled_vol = np.moveaxis(filled_vol, axis, 0)

    for gap in range(1, depth + 1):
        for idx in range(filled_vol.shape[0] - gap - 1):
            fill_charge = np.logical_and(
                filled_vol[idx, ...], filled_vol[idx + gap + 1, ...]
            )
            for f_idx in range(idx + 1, idx + 1 + gap):
                filled_vol[f_idx, ...] = np.logical_or(
                    filled_vol[f_idx, ...], filled_vol[f_idx, ...]
                ).astype(filled_vol.dtype)

    if axis != 0:
        filled_vol = np.moveaxis(filled_vol, 0, axis)

    return filled_vol


def remove_single_slice_objects(vol, axis=None):
    """
    Removes any objects that occupy only a single slice in the volume
    along the axis specified.  Use None or "all" for `axis` to check
    all axes.

    @param      vol           The volume to filter
    @param      axis          Axis to consider (None or "all" to check all)

    @return     The filtered volume is returned.
    """
    vol = np.copy(vol)
    lvol, regions = label(vol, connectivity=2, return_num=True)
    for l_idx in range(1, regions + 1):
        if single_slice_label(lvol, l_idx, axis=axis):
            vol[lvol == l_idx] = 0
    return vol


def single_slice_label(vol, label, axis=0):
    """
    Determines if the given label exists only on a single "slice" along
    `axis`.  Use "all" or None for `axis` to check all axes.

    @param      vol         The labeled image object
    @param      label       Label (value) to check
    @param      axis        Axis to check (None or "all" to check all)
    
    @return     True if `label` exists on only one slice along `axis`.
    """
    if axis is not None and axis != "all":
        result = _single_slice_label(vol, label, axis)
    else:
        result = np.any(
            [_single_slice_label(vol, label, axis=a) for a in np.arange(len(vol.shape))]
        )
    return result


def _single_slice_label(vol, label, axis=0):
    return len(np.unique(np.argwhere(vol == label)[:, axis])) == 1


def remove_small(vol, size=400):
    """
    Removes small objects, specialized to the case of a binary volumentric
    mask.
    
    @param      vol           The volume
    @param      size          The minimum size object allowed to remain
    
    @return     The volume with small objects removed.
    """
    return remove_small_objects(
        vol.astype(bool), min_size=size, connectivity=1, in_place=False
    ).astype(vol.dtype)


def keep_largest(vol, n=2):
    """
    Keeps only the largest `n` connected regions within the volume.
    
    @param      vol   The volume
    @param      n     Number representing how many of the largest connected
                      regions to keep.
    
    @return     A volume equivalent to `vol` with only the `n` largest
                regions retained.
    """
    lvol, regions = label(vol, connectivity=2, return_num=True)
    volumes = []
    for l_idx in range(1, regions + 1):
        volumes.append((l_idx, np.sum(lvol == l_idx)))
    volumes.sort(key=lambda v: v[1], reverse=True)
    vol = np.zeros_like(vol)
    for volume in volumes[:n]:
        l_idx, _ = volume
        vol[lvol == l_idx] = 1
    return vol


def intersect(vol1, vol2):
    """
    Returns the logical intersection of volumes `vol1` and `vol2`.
    """
    return np.logical_and(vol1.astype(bool), vol2.astype(bool)).astype(int)


def fill_objects(vol, axis=0):
    """
    Fills each object by applying a convex hull.  The algorithm
    works slice-by-slice along the specified axis; use None or "all"
    to iterate over all axes.

    @param vol      The volume to examine
    @param axis     Axis to iterate over; use None or "all" for all axes

    @return         The volume with objects filled is returned.
    """
    filled_vol = None
    if axis is not None and axis != "all":
        vol = _fill_objects(vol, axis)
    else:
        for axis in range(len(vol.shape)):
            vol = _fill_objects(vol, axis)
    return vol


def _fill_objects(vol, axis=0):
    filled_vol = np.copy(vol)
    del vol

    if axis != 0:
        filled_vol = np.moveaxis(filled_vol, axis, 0)

    for idx, slc in enumerate(filled_vol):
        slc = convex_hull_object(slc, neighbors=4)
        filled_vol[idx, ...] = slc

    if axis != 0:
        filled_vol = np.moveaxis(filled_vol, 0, axis)

    return filled_vol


def post_process_kt_t(vol, k_steps=[], t_steps=[], kt_steps=[], do_intersect=False):
    """
    Apply post-processing cleanup steps to the combined kidney/tumor volume
    and the tumor-only volume.  Expects `vol` labeled 0,1,2 and returns
    in the same format.
    Updates occur in the order: t_steps, k_steps, (k+t), kt_steps, [intersect]

    @param      vol           The volume
    @param      k_steps       A list of functions to which the K volume will be passed,
                              sequenced in the order they are to be (serially) applied.
    @param      t_steps       A list of functions to which the Tumor volume will be passed,
                              sequenced in the order they are to be (serially) applied.
    @param      kt_steps      A list of functions to which the K+T volume will be passed,
                              sequenced in the order they are to be (serially) applied.
    @param      do_intersect  Set to `True` if you want to force Tumor segmentations to be
                              contained in the intersection of K+T after all post-processing
                              steps are applied to the K+T volume.

    @return     A volumetric 0,1,2 mask equivalent to `vol`, but with all post-processing 
                steps appied.
    """
    vol_tu = (vol == 2).astype("uint8")
    vol_k = (vol == 1).astype("uint8")
    del vol

    for step in k_steps:
        vol_k = step(vol_k)
    for step in t_steps:
        vol_tu = step(vol_tu)

    vol = np.logical_or(vol_tu.astype(bool), vol_k.astype(bool)).astype("uint8")
    del vol_k

    for step in kt_steps:
        vol = step(vol)

    if do_intersect:
        vol_tu = intersect(vol, vol_tu)
    vol[vol_tu > 0] = 2

    return vol


if __name__ == "__main__":
    print("This script is not intended to run stand-alone", file=sys.stderr)
    sys.exit(1)
