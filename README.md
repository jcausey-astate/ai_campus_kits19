# KiTS 2019 Models from AI-Campus
## Introduction
The [2019 Kidney Tumor Segmentation Challenge](https://kits19.grand-challenge.org/) provide a good platform for encouraging computational approach development for automatic kidney tumor segmentation with patients CT scans. In this manuscript we provide our method to address the challenging question. Our method is based on neural network models and trained by the dataset provided by the KiTS19 Challenge.

The data were provided by the KiTS19 Challenge organization[@heller2019kits19]. There are 300 patients of CT data. 210 patients data were made available to the competition teams, and the remaining 90 patients data are used for testing predictions; no segmentation information was provided for these.

We used the 210 patient CT scans corresponding ground truth provided by the KiTS19 Challenge organizers[@heller2019kits19] for our training and validation.  Our validation group was set aside before training began by selecting 20% (N=42) of available patients at random.  The same validation group was isolated and used for validation on all models.  The remaining 168 patients were used as our training group for all models.

We decided on an ensemble of U-Net models as our final configuration after testing many variations.  We discuss our experience with Mask-RCNN further in the Discussion section, as well as our rationale for ultimately choosing U-Net.

Our final ensemble consists of two U-Net models working in tandem, followed by a post-processing "cleanup" phase to minimize prediction artifacts.  Both models in our ensemble were trained on axial slices, differing in the number of epochs trained and the interpretation of the output masks from each.  One model was tasked with predicting the kidney and tumor masks separately in its two output channels.  We will refer to this as the "K/T" model.  The other model was trained to predict the combined kidney+tumor mask on the first output channel, and the tumor portion on the second output channel.  We will refer to this as the "KT/T" model.  The output from the two models was combined such that both models voted equally for the inclusion of any individual mask voxel, and voxels receiving a vote from either model were included in the result sent to the post-processing stage.

## Python Environment
We recommend using [Pipenv](https://github.com/pypa/pipenv) to manage the Python environment.  A Pipfile is included in this repository.

## Pre-processing
We found that loading the NiFTi-format files for each patient was a bottleneck in our training process, so we pre-processed the images and saved the pre-processed versions in a format that could be read directly by the Numpy package.  For our axial models, we saved each axial "slice" in an individual Numpy file.  This allowed us to load slices individually instead of loading an entire CT scan volume, further optimizing our loading times.  For training with coronal and saggital views, we saved the entire CT volume for each patient in a single Numpy file.  We optimized training on these views such that all possible slices for a single patient were used preferentially before moving to a different patient, so that we could reduce the impact of the longer load times.

Our pre-processing also included a window normalization of the CT image data which thresholded the raw Hounsfield units to the range $[-500,500]$ and mapped the values to the numeric range $[0,1]$ according to the formula:
$$
v_{out} = \mathrm{min}( \mathrm{max}(v_{in}, -500), 500 ) \cdot \frac{1}{1000} + 0.5
$$
This step must also be performed prior to inference with all our models, so it is part of the input stage for the inference algorithm.

For inference, our algorithm reads the NiFTi file directly; it is not necessary to cache the image in Numpy format at this stage.  The window normalization step is required as a pre-processing step during inference.

## Training

Our training data consisted of 168 cases that included a ground-truth segmentation.  For each of our models, we proceeded as follows using the Keras[Keras] deep learning framework in Python with the Tensorflow[Tensorflow] back-end.

1. Starting weights were "seeded" at random and trained for 8 epochs each.  We continued this process until we found an initial model that seemed to be converging at a reasonable rate.  Many starts did not converge in any meaningful way within the first 8 epochs, and were discarded.  In general, a good starting weights could be found in about 5 attempts.
2. All training for the axial models proceeded by dividing all available axial slices into two sets: "Positive" slices contained at least one segmented voxel of either tumor or kidney, and "Negative" slices contained no segmented voxels.  We balanced our training set by randomly choosing enough slices from the positive and negative sets to create a 2:1 ratio of positive slices.
3. Image slices were augmented in the following ways (each augmentation had a 50% chance of being applied to any slice):
    * Randomly flipped vertically   (this augmentation was disabled after ~135 epochs)
    * Randomly flipped horizontally
    * Randomly shifted up to 15% in both the vertical and horizontal directions
    * Randomly zoomed in/out up to 15% and re-cropped or padded with zeros to maintain image size  (only used on epochs > 150 K/T and > 200 KT/T)
4. Models were trained using \~2000 slices per epoch.  Training loss was a weighted cross-entropy loss where tumor segmentation errors were weighted 10x versus kidney segmentation errors.  We also monitored a per-slice DICE metric to determine how training was proceeding.
5. After training the models until the training metrics indicated a performance plateau, we ranked the weights by training and validation dice metric, and chose several top ranked checkpoints for further testing.  For both axial models, we eventually trained in excess of 250 epochs, but the later checkpoints were not always best.

Selected "best" weights were then used in an ensemble as described prevously; we chose one checkpoint from the K/T and KT/T models for our final ensemble.

### Step-by-Step Training:
First, edit `train_2d_multi_view_generator.py` if desired for batch size default and augmentation steps (augmentation can be modified just by commenting or un-commenting lines in the `__simple_aug()` methods).

Run either `run_axial.sh` or `run_axial_kt-t.sh` to produce the seed weights.  This will generate five attempts at seed weights; we found that the axial models usually produce a seed that is showing good training progress within 5 attempts --- but since this is random, more attempts may be required (re-run the script).

Using the best seed weights, continue training for the desired number of epochs using the `--resume-training` and `--resume-from-epoch` options in the `train_2d_multi_view_generator.py` script.

You can use the `ensemble_unet_multiple_tests_with_post_process.py` script to evaluate your weight checkpoints (with different combinations of post-processing and ensemble options).  This does require editing the script to set up the desired models and weights; commented-out example code is in place to help.

## Inference

The `ensemble_unet_multiple_tests_with_post_process.py` script can be used to score a validation set against ground-truth masks, given a set of saved model weights. This does require editing the script to set up the desired models and weights; commented-out example code is in place to help.

To produce prediction volumes on unknowns (without a ground truth), use one of the `run_predictions_for_submission_X.py` scripts (where `X` is a number).  Or, if you have trained new weights, use one of those scripts as a starting point and edit the model weights filename to match your own weights.

