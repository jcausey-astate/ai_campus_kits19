"""
Defines a common function for predicting with the multiview model; use this
with `orchestra` to build an ensemble.
"""

import os, sys
import numpy as np
import unet
import tensorflow as tf
from keras import backend as K
from kits_volume_utils import *
from keras.models import load_model
from tensorflow import Graph
from tensorflow import Session
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

BATCH_SIZE = 14

def predict_multiview(img_vol, weights_file, orientation='axial', tu_thresh=0.1, k_thresh=0.2, tch_only=False):
    """
    Given a volume `vol` and the weights to use, along with the 
    desired orientation, predict an output volume and return it.
    """

    model = unet.get_unet() if not tch_only else unet.get_unet_Tch()
    model.load_weights(weights_file) # load weights

    # Transform the volume to the correct orientation.
    img_vol = reorient_volume(img_vol, orientation)

    # re-size so that we have 512x512 along the z-axis:
    orig_shape = img_vol.shape
    img_vol    = resize_volume_slices(img_vol, target_shape=512)
    pred_vol   = np.zeros(img_vol.shape, dtype='uint8')
    
    if len(img_vol.shape) < 4:
        img_vol = np.expand_dims(img_vol, axis=3)

    # Now we have the image and seg volumes.  Predict in batches:
    n_batches = int(np.ceil(img_vol.shape[0] / BATCH_SIZE))
    # Batch up the slices and feed them through.
    min_pred, max_pred = [], []
    for batch_idx in range(n_batches):
        # print("    Processing batch {} of {}.".format(batch_idx+1, n_batches))
        min_idx = batch_idx * BATCH_SIZE
        max_idx = min((batch_idx + 1) * BATCH_SIZE, img_vol.shape[0])
        pred_batch = model.predict(img_vol[min_idx:max_idx,...])
        min_pred.append(pred_batch.min())
        max_pred.append(pred_batch.max())
        if not tch_only:
            pred_batch_i = np.greater(pred_batch[..., 0], k_thresh).astype('uint8')
            pred_batch_i[np.greater(pred_batch[..., 1], tu_thresh)] = 2
        else:
            pred_batch_i = np.greater(pred_batch[..., 0], tu_thresh).astype('uint8') * 2
        pred_vol[min_idx:max_idx,...] = pred_batch_i
    
    # Reshape to match the original:
    pred_vol = restore_volume_to_shape(pred_vol, orig_shape, is_mask=True)
    # And return to the original orientation:
    pred_vol = reorient_to_axial(pred_vol, orientation)
    
    return pred_vol

