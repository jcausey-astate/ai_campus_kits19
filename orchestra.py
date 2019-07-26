"""
orchestra.py:  This job will coordinate an ensemble model between an arbitrary list of models.
Author:  Jonathan Stubblefield
Edits by: Jason L Causey
Date:  7/5/2019
"""

import pathlib
import nibabel
import numpy
import os

class Orchestra:
    def __init__(self, models, weights, datapath, outpath, unity_weights=True, post_process=None):
        #Arguments:
        #models - A list of callable objects.  These objects should accept a 3-D numpy array containing a CT-scan and return a mask of kidney and tumor of the same shape.
        #    kidney = 1, tumor = 2
        #
        #weights - A list of tuples.  Each tuple expected to contain two floats:  (kidney weight, tumor weight).  These will be used to weight the individual models and are assumed to add to 1.
        #
        #datapath - Root directory of the dataset.
        #
        #outpath - A destination for the results to be stored in the form of nifti/gz files.
        #
        #unity_weights - If true, all weights will be normalized to add to 1.

        assert(len(models) == len(weights))
        
        self.models = models
        self.post_process = post_process

        weights = numpy.asarray(weights)
        t_weights = numpy.asarray([t for k, t in weights])
        k_weights = numpy.asarray([k for k, t in weights])

        if(unity_weights):
            t_weights = t_weights / float(t_weights.sum())
            k_weights = k_weights / float(k_weights.sum())

        self.weights = numpy.asarray([(k_weights[i], t_weights[i]) for i in range(len(t_weights))])

        self.datapath = pathlib.Path(datapath)
        self.outpath = pathlib.Path(outpath)
        if not self.outpath.is_dir():
            os.makedirs(str(self.outpath))

        #Configurable Parameters:
        self.HU_max = 500
        self.HU_min = -500

        self.k_thresh = 0.5
        self.t_thresh = 0.5

    def predict(self, cases):
        #Arguments:
        #cases - A list of ints designating which cases to predict on.
        
        #Predict on specified cases.
        for caseid in cases:
            if str(caseid).lower().startswith('case'):
                casename = caseid
                caseid = int(caseid.split("_")[1])
            else:
                caseid = int(caseid)
                casename = 'case_{:05d}'.format(caseid)
            outname = 'prediction_{:05d}.nii.gz'.format(caseid)
            img_file = self.datapath / casename / 'imaging.nii.gz'
            
            print('Analyzing ' + casename + '.')

            #Call function for predicting on this case.
            volume, affine = self.load_nifti(str(img_file.absolute()))
            mask = self.analyze_volume(volume)
            if self.post_process is not None:
                print('Post-processing.')
                mask = self.post_process(mask)
            self.save_mask(outname, mask, affine)

    def load_nifti(self, path, is_mask=False):
        print('Loading {}.'.format('case' if not is_mask else 'segmentation'))
        volume = nibabel.load(path)
        affine = volume.affine
        volume = volume.get_fdata()
        
        if not is_mask:
            #Next, standardize Hounsfield units.
            print('Standardizing Hounsfield Units.')
            volume = self.standardize_HU(volume)

        return volume, affine

    def save_mask(self, name, final_mask, affine):
        #Lastly, we need to save this result in the output path.
        #Get filename.
        destination = str((self.outpath / name).absolute())

        #Save the nifti.
        mask = nibabel.Nifti1Image(final_mask, affine)
        nibabel.save(mask, destination)

    def analyze_volume(self, volume, use_post_proc=False):
        #Analyze this single case.
        #Create destination matrices.
        kidney_mask = numpy.zeros(volume.shape, dtype='float16')
        tumor_mask = numpy.zeros(volume.shape, dtype='float16')

        #Predict using each model, one at a time.
        for i in range(len(self.models)):
            print('Prediction with model ' + str(i) + '.')
            
            prediction_k = self.models[i](volume)

            #Make sure shape is correct and prediction is dtype int8.
            assert(prediction_k.shape == volume.shape)

            #Split mask into tumor and kidney parts.
            prediction_t = numpy.zeros(prediction_k.shape, dtype='int8')
            prediction_t[prediction_k == 2] = 1
            prediction_k[prediction_k == 2] = 0

            #Multiply each result matrix by this model's weight.
            prediction_k = (prediction_k * self.weights[i][0]).astype('float16')
            prediction_t = (prediction_t * self.weights[i][1]).astype('float16')

            #Add the results to our destination matrices.
            kidney_mask = kidney_mask + prediction_k
            tumor_mask = tumor_mask + prediction_t

            #Manually clean up.
            del prediction_k
            del prediction_t

        #Now, we have accumulated the model's votes.  Threshold.
        print('Thresholding final mask.')
        
        final_mask = numpy.zeros(volume.shape, dtype='int8')
        final_mask[kidney_mask >= self.k_thresh] = 1
        final_mask[tumor_mask >= self.t_thresh] = 2

        if use_post_proc and (self.post_process is not None):
            final_mask = self.post_process(final_mask)

        return final_mask

    def standardize_HU(self, vol):
        vol[vol > self.HU_max] = self.HU_max
        vol[vol < self.HU_min] = self.HU_min

        conversion_factor = 1.0 / (self.HU_max - self.HU_min)
        conversion_intercept = 0.5
        vol = vol * conversion_factor + conversion_intercept

        assert numpy.amax(vol) <= 1, "Max above one after normalization."
        assert numpy.amin(vol) >= 0, "Min below zero after normalization."

        return vol
    
