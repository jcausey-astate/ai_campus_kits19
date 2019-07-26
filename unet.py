"""
Defines the structure of the U-net.
"""
from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Lambda,add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

size= 512
channel=1
batch_size=16
ss=0.5


def ccc_crossentropy_cat(y_true,y_pred,cat):

    y_true_f = K.flatten(y_true[:,:,:,cat])
    y_pred_f = K.flatten(y_pred[:,:,:,cat])
    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1.- 1e-7))
    out = -(y_true_f * K.log(y_pred_f)+ (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    out=K.mean(out)
    return out

def ccc_crossentropy(y_true,y_pred):

    a = ccc_crossentropy_cat(y_true,y_pred,0)
    b = ccc_crossentropy_cat(y_true,y_pred,1)
    final=(a+b*10)/2.0
    return final

def dice_coef_cat(y_true, y_pred, cat, smooth=0.01):
    y_true_f = K.flatten(y_true[:,:,:,cat])
    y_pred_f = K.flatten(y_pred[:,:,:,cat])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean(((2. * intersect + smooth)  / (denom + smooth)))

def dice_coef(y_true, y_pred, smooth=1):
    a=dice_coef_cat(y_true,y_pred,0)
    b=dice_coef_cat(y_true,y_pred,1)
    #c=dice_coef_cat(y_true,y_pred,2)
    #final=(a+b+c)/3.0
    final=(a+b)/2.0
    return final

def dice_coef_weighted(y_true, y_pred, smooth=1):
    a=dice_coef_cat(y_true,y_pred,0)
    b=dice_coef_cat(y_true,y_pred,1)
    final=(a+2*b)/3.0
    return final


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def dice_loss_weighted(y_true, y_pred):
    return 1.0 - dice_coef_weighted(y_true, y_pred)


def get_unet_core(omit_last=False):
    inputs = Input((size, size, channel))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(2, (1, 1), activation=('sigmoid'), name='conv10')(conv9)
    #conv10 = Conv2D(3, (1, 1), activation=('softmax'), name='conv10')(conv9)
    return inputs, conv10 if not omit_last else conv9

def get_unet():
    
    inputs, final = get_unet_core()
    model = Model(inputs=[inputs], outputs=[final])
    #model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, decay=0.0),  loss= 'categorical_crossentropy', metrics=[dice_coef])

    #model.compile(optimizer=Adam(lr=3e-5,beta_1=0.9, beta_2=0.999, decay=0.0), loss=dice_coef_loss, metrics=[dice_coef_loss])
    model.compile(optimizer=Adam(lr=3e-5,beta_1=0.9, beta_2=0.999, decay=0.0), loss=ccc_crossentropy, metrics=[dice_coef_loss])
    return model
