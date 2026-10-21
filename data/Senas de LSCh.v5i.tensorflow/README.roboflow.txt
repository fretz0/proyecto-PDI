
Señas de LSCh - v5 2024-08-12 2:31pm
==============================

This dataset was exported via roboflow.com on April 27, 2025 at 10:22 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2660 images.
Letra are annotated in Tensorflow Object Detection format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -5 and +5 degrees
* Random brigthness adjustment of between -25 and +25 percent

The following transformations were applied to the bounding boxes of each image:
* Random shear of between -5° to +5° horizontally and -5° to +5° vertically
* Random Gaussian blur of between 0 and 1.2 pixels


