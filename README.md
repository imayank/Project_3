## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The aim of the project is to build a traffic sign classifier. While driving the vehicle around it is very important to lookout for various traffic signs as the signs provide important information about the upcoming scenario on the road (or about the traffic in general) and a human driver act according to the information gained. Hence, it is very important for a self-driving vehicle to locate and recognize the type of the traffic sign. This project aims on the recongnition part of the task and builds a traffic sign classifier using Convolutional Neural Network. The model is trained and validated on [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Finally the model is tested on the new German traffic sign images obtained on web.

The Project
---
The steps of this project are the following:
* Loading the German traffic signs dataset
* Exploratory data visualization
* Designing, training and validating a Convolutional Neural network model
* Testing the model on new images obtained online.
* Analyzing the softmax probabilities for top 5 picks of the classifier
* Writing a project writeup.

### Dependencies
To successfully run this project following python dependencies are needed:
* numPy
* matplotlib
* sklearn
* tensorflow
* keras

### Dataset
The dataset is strored as pickled files. The dataset is pre-divided into training set, validation set and test set and the respective pickled file are as follows *train.p, valid.p, test.p.* Every image in the dataset has been re-sized to 32x32 pixels. More details about the dataset can be found in Exploratory Data Analysis section of the Ipython notebook - *Traffic_sign_classifier.ipynb*

### Main files and folders
* `Traffic_sign_classifier.ipynb` is the Ipython notebook containing the complete code of the project.
* `writeup.md` is the main project writeup explaining the various steps followed in the project. **Great place to start** 
* `new_signs` folder contains new images on which the classifier was tested.
* `output_images` contains images used in *writeup.md*
* Other files are mainly saved models.


