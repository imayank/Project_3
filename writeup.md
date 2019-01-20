# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/random_visualization.jpg "Random_Visualization"
[image2]: ./output_images/class_distribution.jpg "Visualization"
[image3]: ./output_images/training_distribution.jpg "training_Visualization"
[image4]: ./output_images/colored.jpg "Colored_set"
[image5]: ./output_images/grayscale.jpg "Grayscaled_set"
[image6]: ./output_images/augmented.jpg "Augmented"
[image7]: ./output_images/augmented_training_distribution.jpg "Augmented_distribution"
[image8]: ./output_images/normalized.jpg "normalized"
[image9]: ./output_images/VGG.png "VGG-16"
[image10]: ./new_signs/r1.jpg "New_image1"
[image11]: ./new_signs/r2.jpg "New_image2"
[image12]: ./new_signs/r3.jpg "New_image3"
[image13]: ./new_signs/r4.jpg "New_image4"
[image14]: ./new_signs/r5.jpg "New_image5"
[image15]: ./new_signs/r6.jpg "New_image6"
[image16]: ./output_images/top_predictions.jpg "Top_predictions"



#### 1. Link to Project notebook

Here is a link to my [project notebook](https://github.com/imayank/Project_3/blob/master/Traffic_sign_classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Dataset Summary

I used the numpy library to calculate summary statistics of the traffic signs data set. The code can be found in the notebook cell titled **Basic Summary** 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory Data Analysis

Here is an exploratory visualization of the data set. The respective code cells in the notebook are titled **Exploratory Data Analysis**.

##### Visualizing some random images from the training set

Let's begin by visualizing some random images from the training set, this will give some idea about the content of training set. Below is presented some of the images from training set:

![alt text][image1]

This visualization gives this idea that the images are not that much clear and some images are also taken in low light.

##### Distribution of classes in training set , validation set and test set

Below is presented the bar chart depicting the distribution of classes in training, validation and test set respectively. The distribution of classes in each of the mentioned set is similar. The similar distribuition of classes is a good thing when validating and testing the model.

![alt text][image2]


##### Class distribution in training set and data augmentation

It can be seen in the plot presented below that number of training examples in different classes are unequal. Some classes have very less examples then other classes, let's call them minority classes. These minority classes might be under-represented when training the model on this training set. Hence, it is a good idea to augment the image data for these minority classes.

![alt text][image3]


### Design and Test a Model Architecture

##### Conversion to grayscale

As first step of preprocessing, all the images were coverted to grayscale. The reason behind this was that no new information was being added by the color in the traffic sign image. Color was not the mazorly distinguishing among traffic signs. Also, some pictures that were taken in bad light conditions were appearing clearer when converted to grayscale. Secondly, converting to grayscale would meant faster training of the model.

The respective code is contained in the notebook under the section titled **Grayscaling**

Here is an example of a traffic sign images before and after grayscaling.

![alt text][image4]

![alt text][image5]


##### Image Data Augmentation

The respective code is contained in the notebook under the section titled **Data Augmentation**

As stated above some of the classes are under-represented in the training data and termed them as miniority classes of our training data. This under-representation might lead to poor model. The solution to the problem is generating additional data for these classes.

The following steps were followed in selecting the classes for augmentation and augmenting the data:

* All classes having size less than 800 were selected as minority classes for data augmentation. The mean of size of all classes is 809.27, the number 800 was chosen with respect to the mean value.

* For data augmentation **ImageDataGenerator** from **keras** is used.
  ```
  from keras.preprocessing.image import ImageDataGenerator
  datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        featurewise_center=False,
        featurewise_std_normalization=False
        )
  ### This data generator randomly rotates or shifts or shears image in the provided range and generate the output image.
  ```
  This data generator is used to generate batches of randomly transformed images to augment the data in each of the minority class till the size of the class is at least 800.
  
Some of the examples of transformed images are shown: 

![alt text][image6]


The distribution of classes in the training set after augmentation step is presented below:

![alt text][image7]

##### Image Normalization

The respective code is contained in the notebook under the section titled **Normalization**

As next step in image pre-processing, images were normalized to have zero mean and equal variance. The below formula is used for image normalization:

```
  new_pixel_value = (pixel_value-128)/128
```

The nomalization step will make data center rougly around zero and in roughly equal range. Due to this the optimizer will converge faaster to better results. An exaple comparison is given below:

![alt text][image8]


##### Shuffling

The respective code is contained in the notebook under the section titled **Shuffliing**

As last step in pre-processing the training data is shuffled.

```
from sklearn.utils import shuffle

X_train_shuffled, y_train_shuffled = shuffle(X_train_normalized, y_train)
```



#### 2. Final Model Architecture

The respective code is contained in the notebook under the section titled **Model Architecture**

My final model consisted of the following layers:

| Layer                    |     Description                               | 
|:------------------------:|:---------------------------------------------:| 
| Input                    | 32x32x3 RGB image                             | 
| layer_1: Convolution 3x3 | 1x1 stride, SAME padding, outputs 32x32x64    |
| layer_1: ReLU            | Activation Layer                              |
| layer_1: Max pooling 2x2 | 2x2 stride,  outputs 16x16x64                 |
| layer_2: Convolution 3x3 | 1x1 stride, SAME padding, outputs 16x16x128   |
| layer_2: ReLU            | Activation Layer                              |
| layer_2: Max pooling 2x2 | 2x2 stride, outputs 8x8x128                   |
| layer_3: Convolution 3x3 | 1x1 stride, SAME padding, outputs 8x8x256     |
| layer_3: ReLU            | Activation Layer                              |
| layer_3: Max pooling 2x2 | 2x2 stride, outputs 4x4x256                   |
| layer_4: Convolution 3x3 | 1x1 stride, SAME padding, outputs 4x4x512     |
| layer_4: ReLU            | Activation Layer                              |
| layer_4: Max pooling 2x2 | 2x2 stride, outputs 2x2x512                   |
| flatten_layer: FLATTEN   | Flattening output of layer_4, 2x2x512 -> 2048 |
| flatten_layer: DROPOUT   | Dropout layer                                 | 
| layer_5: FULLY CONNECTED | Fully connected layer of size 1024, outputs 1024x1 |
| layer_5: ReLU            | Activation Layer                              |
| layer_5: DROPOUT         | Dropout layer                                 |   
| logits: FULLY CONNECTED  | Fully connected layer of size 43, outputs 43x1|
 


#### 3. Training parameters and hyperparameters

Following setting was used for training the model:

* **Optimizer:** Adam Optimizer
* **Learning Rate:** 0.001
* **Loss Functiom:** Cross entropy
* **EPOCHS:** 15
* **keep probability(keep_prob):** When training: 0.5
* **BATCH SIZE:** 128
* **tf.truncated_normal():** mean = 0, and standard deviation = 0.1 

#### 4. Approach towards solution

The respective code is contained in the notebook under the section titled **Train, Validate and Test the Model**

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.967 
* test set accuracy of 0.948

The architecture of the model is based on one of the popular CNN architecture VGG-16. VGG-16 is a neural network architecture that performed really well in 2014's Imagenet competition. Because it performed so well on Imagenet database, I thought that an architecture based on VGG-16 might perform well for a task like traffic sign classifiction also. The second reason to base my model on VGG-16 was its simplicity.

VGG-16 uses 224x224 RGB image as input, and keep reducing its height and width while increasing the depth till it obtains a feature stack of shape 7x7x512. Which is followed by 2 fully  connected layer and finally a softmax layer. It is shown in the image below:

![alt text][image9]


Because the size of the traffic sign images is 32x32x1, the architecture for classifying traffix signs cannot be as deep as original VGG-16. So, I have made following changes to the original architecture:

1. In the  VGG-16 architecture there are two convolutional layers followed by a max pooloing layer. In the present architecture only one convolutional layer was created followed by a max pooling layer.

1. The last set of convolutional+convolutional+maxpool layer is removed in present architecture.

1. Dropout layers were added so as the model does not gets overfitted over the training data.

After the model architecture was chosen, and testing that it was working well different combinations of *keep_prob* , *learning_rate* and *EPOCHS* were tried. The best parameters were chosen that have been already reported above.


### Test a Model on New Images

The respective code is contained in the notebook under the section titled **Test a model on new Images**

Here are German traffic signs that I found on the web:

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14] ![alt text][image15]

The images I found on the web are vbright and clear, and there is nothing that should make classifying them difficult. The new images I found are more distinguishable than the images in the training set. That's why I expect all of them to be classified with high confidence. 

#### 2. Model's prediction on new images

Here are the results of the prediction:

| Image                                 |     Prediction                                | 
|:-------------------------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection | Right-of-way at the next intersection         | 
| Speed limit (60km/h)                  | Speed limit (60km/h)                          |
| No passing                            | No passing                                    |
| Traffic signals                       | Traffic signals                               |
| Keep right                            | Keep right                                    |
| Road work                             | Road work                                     |



The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.8%. As all the images were clear and bright I expected 100% accurate predictions on the new traffic sign images. It can also be concluded that if the real world images of traffic signs (taken from car cameras) are as clear as these the accuracy of the model will remain high, as in this case.

#### 3. Classifier confidence

The respective code is contained in the notebook under the section titled **Top Predictions**

The Input images and top 5 predications with probabilities is presented in the image below:

![alt text][image16]

