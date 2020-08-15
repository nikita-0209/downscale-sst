# Downscaling Oceanographic Satellite Data with Convolutional Neural Networks

A widely measured variable in the ocean, Sea SurfaceTemperature  (SST),  is  a  strong  indicator  of  pollution, productivity, global climate change and stress to corals and other species.  It is an estimate of the energy in the sea due to the motion of molecules. High resolution satellite sensors are effectively measure Sea Surface Temperature under clear sky conditions.  However under cloudy conditions, high resolution SST Measurements are not available.With  the  help  of  a  deep  learning  architecture,  the available images with low spatial resolution can beenhanced to produce images of high spatial resolution.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

* Python3
* Tensorflow 2.x
* NetCDF
* Basemaps Toolkit

### Installing

After ensuring you have the above mentioned versions of Tensorflow and Python. The other two prerequisties can be installed in a Colab notebook as:

NetCDF: 
```
!pip install netCDF4
```

And 

```
!apt-get install libgeos-3.5.0
!apt-get install libgeos-dev
!pip install https://github.com/matplotlib/basemap/archive/master.zip
```

## The Data

Group of High Resolution Sea Surface Temperature (GHRSST) data engulf SST observations from all kinds of available sources. The  GHRSST was established to foster an international focus and coordination for the development of a new generation of global, multi-sensor, high-resolution near real time SST datasets. Major contribution in this dataset comes from the space-borne satellite radiometers. The Level-4 (L4) product is generated using various objective analysis techniques to produce gap-free SST maps over the global oceans. In this study, we have used this L4 GHRSST products with a regular spatial resolution of ~ 1 km. The data was downloaded from [Physical Oceanography Distributed Active Archive Center,](https://podaac.jpl.nasa.gov/GHRSST)

## Methodology

### Create Data

Sea Surface Temperature Data is stored in NetCDF Format. Along with recording the sea surface temperature, these data files denote land values by a particular constant. This constant, known as the fix value differs from one data file to another. To maintain uniformity, all NetCDF files were rewritten to assign a single fix value, (in this case 0) to denote the land values. For purposes of training, each of the given SST fields were divided into overlapping patches. 
This can be done by [Create Dataset.ipynb](https://github.com/nikita-0209/downsample-sst/blob/master/Create_Dataset.ipynb). Remember to change the paths to where your NetCDF Files are stored. 
As required by the architecture, the images were normalized to range [0,1] by dividing each pixel with the maximum of pixel values in both the data files.

### Models

Since Super Resolution Convolutional Neural Network had already been tried and tested on bicubic interpolated SST Fields by Aurelien Ducournau and Ronan Fablet in their paper Deep Learning for Ocean Remote Sensing: An Application of Convolutional Neural Networks for Super-Resolution on Satellite-Derived SST Data, initial experiments were carried out on this architecture. Since the results weren't satisfactory, a deeper architecture, namely Very Deep Super Resolution CNN was experimented with. For both the variants, the activation function used is ReLu. Each model is optimised by adaptive moment estimation. A batch size of 64 was chosen. 

To run the SRCNN Architecture:
```
python srcnn_server.py --file_name_low <path to hdf5 array of low resolution patches>  --file_name_high  <path to hdf5 array of high resolution patches>
```

To run the VDSR Architecture:
```
python vdsr_server.py --file_name_low <path to hdf5 array of low resolution patches>  --file_name_high  <path to hdf5 array of high resolution patches>
```

The checkpoints of the best models will be saved in ckpts directory. Currently the number of epochs is 100.

## Evaluate

Each model was trained to minimise the mean square error between the predicted and the expected patch. Along with that, a popular metric used for comparing quality of images, Peak Signal Noise Ratio (PSNR) was calculated. The smaller the MSE, the greater is the PSNR and the better is the image quality.
[Evaluate.ipynb](https://github.com/nikita-0209/downsample-sst/blob/master/Evaluate.ipynb) initialises the model, loads the weights and calcuulates PSNR of the predicted patches.

## Predictions

To reconstruct the entire SST Field from the predicted patches, run [Prediction.ipynb](https://github.com/nikita-0209/downsample-sst/blob/master/Prediction.ipynb). This model assumes that VDSR was trained. Feel free to replace the architecture if needed.
The model is initialised, the weights are loaded, each patch is predicted and appropriately arranged to form the final complete SST Field.

Remember to change paths to the saved hdf5 files of patches and model weights. 
Modify the path while writing the created NetCDF File of Predicted SST Fields.

##  Evaluation Results

In order to gain a better insight into the performance of the model, several fields like Mean of Predicted SST Fields, Domain Averaged Root Mean Square Error (RMSE) and Domain Averaged Bias have been plotted in [Plot Evaluation.ipynb](https://github.com/nikita-0209/downsample-sst/blob/master/Plot_Evaluation.ipynb). The terms Mean SST and Domain Averaged RMSE are self-explanatory with mean SST referring to the mathematical mean of the predicted SST Fields, and the latter referring to the averaged root mean squared error between considered fields. 
The Domain Averaged Bias is defined as the mean of the differences between the considered SST Fields, say the input and the predicted.

## Detailed Report

A detailed report of this project is available: [here](https://drive.google.com/file/d/1ssvq1EZvxojmPIaApwvClPOaoqPUlmu8/view?usp=sharing).

## Authors

* **Nikita Saxena** 

## Acknowledgments
I express my sincere thanks to Dr. Rashmi Sharma, who provided me with the opportunity to work on this project. I pay my deep sense of gratitude to Dr. Neeraj Agarwal and Dr. Jishad M, without whose valuable guidance and supervision the project couldn't have been completed.


