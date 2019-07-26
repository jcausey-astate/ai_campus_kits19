"""
defines scoring functions for checking dice score on Kidney and Tumor 
segmentation volumes.
"""
import numpy as np
import keras.backend as K

def kt_dice_score(true_mask, pred_mask):
    if not np.issubdtype(pred_mask.dtype, np.integer):
        raise RuntimeError("Predictions must be integers with 0,1,2 labels for kt_dice_score.")
    gt   = np.greater(true_mask, 0)
    pred = np.greater(pred_mask, 0)
    kt_dice = dice_score(gt, pred)

    gt   = np.greater(true_mask, 1)
    pred = np.greater(pred_mask, 1)
    tu_dice = dice_score(gt, pred)

    return kt_dice, tu_dice

def dice_score(true_mask, pred_mask):
    try:
        # Compute single-class Dice
        pd = np.greater(pred_mask, 0)
        gt = np.greater(true_mask, 0)
        dice = (2. * np.logical_and(pd, gt).sum()) / (
            pd.sum() + gt.sum()
        )
    except ZeroDivisionError:
        return 0.0
    return dice

def dice_coef(y_true, y_pred, smooth=1):
    non_batch_axes = [1,2,3]
    intersection = K.sum(y_true * y_pred, axis=non_batch_axes)
    union = K.sum(y_true, axis=non_batch_axes) + K.sum(y_pred, axis=non_batch_axes)
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(true_mask, pred_mask):
    return 1.0 - (dice_coef(true_mask, pred_mask)) # negated dice coef makes a loss we can use
