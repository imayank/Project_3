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
[image9]: ./output_images/transformed_21.jpg "Transform_0"
[image10]: ./output_images/transformed_40.jpg "Transform_0"
[image11]: ./output_images/transformed_41.jpg "Transform_0"
[image12]: ./output_images/transformed_42.jpg "Transform_0"
[image13]: ./output_images/transformed_33.jpg "Transform_0"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/imayank/Project_3/blob/master/Traffic_Sign_Classifier_with_Augmentation.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

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

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


##### Conversion to grayscale

As first step of preprocessing, all the images were coverted to grayscale. The reason behind this was that no new information was being added by the color in the traffic sign image. Color was not the mazorly distinguishing among traffic signs. Also, some pictures that were taken in bad light conditions were appearing clearer when converted to grayscale. Secondly, converting to grayscale would meant faster training of the model.

Here is an example of a traffic sign images before and after grayscaling.

![alt text][image4]

![alt text][image5]


##### Image Data Augmentation

As stated above some of the classes are under-represented in the training data and termed them as miniority classes of our training data. This under-representation might lead to poor model. The solution to the problem is generating additional data for these classes.

The following steps were followed in selecting the classes for augmentation and augmenting the data:

* All classes having size less than 800 were selected as minority classes for data augmentation. The mean of size of all classes is 809.27, the number 800 was chosen with respect to the mean value.

* For data augmentation **ImageDataGenerator** is used.
  ```
  from keras.preprocessing.image import ImageDataGenerator
  datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='constant',
        cval=0
        )
  ```
  This data generator is used to generate batches of randomly transformed images to augment the data in each of the minority class till the size of the class is at least 800.
  
Some of the examples of transformed images are shown: 

![alt text][image6]


The distribution of classes in the training set after augmentation step is presented below:

![alt text][image7]

##### Image Normalization

As next step in image pre-processing, images were normalized to have zero mean and equal variance. The below formula is used for image normalization:

```
  new_pixel_value = (pixel_value-128)/128
```

The nomalization step will make data center rougly around zero and in roughly equal range. Due to this the optimizer will converge faaster to better results. An exaple comparison is given below:

![alt text][image8]


##### Shuffling

As last step in pre-processing the training data is shuffled.

```
from sklearn.utils import shuffle

X_train_shuffled, y_train_shuffled = shuffle(X_train_normalized, y_train)
```



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                    |     Description                               | 
|:------------------------:|:---------------------------------------------:| 
| Input                    | 32x32x3 RGB image                             | 
| layer_1: Convolution 5x5 | 1x1 stride, VALID padding, outputs 28x28x6    |
| layer_1: ReLU            | Activation Layer                              |
| layer_1: Max pooling 2x2 | 2x2 stride,  outputs 14x14x6                  |
| layer_2: Convolution 5x5 | 1x1 stride, VALID padding, outputs 10x10x16   |
| layer_2: ReLU            | Activation Layer                              |
| layer_2: Max pooling 2x2 | 2x2 stride, outputs 5x5x16                    |
| layer_3: Convolution 5x5 | 1x1 stride, VALID padding, outputs 1x1x400    |
| layer_3: ReLU            | Activation Layer                              |
| flatten_layer_2: FLATTEN | Flattening output of layer_2, 5x5x16 -> 400   |
| flatten_layer_3: FLATTEN | Flattening output of layer_3, 1x1x400 -> 400  |
| concat: CONCAT           | concatenate flatten_layer_2 & flatten_layer_3, (400,400) -> 800 |
| concat: DROPOUT          | Dropout layer                                 |
| layer_4: FULLY CONNECTED | Fully connected layer of size 120, outputs 120x1 |
| layer_4: ReLU            | Activation Layer                              |
| layer_4: DROPOUT         | Dropout layer                                 |   
| layer_5: FULLY CONNECTED | Fully connected layer of size 84, outputs 84x1|
| layer_5: ReLU            | Activation Layer                              |
| layer_5: DROPOUT         | Dropout layer                                 |
| logits: FULLY CONNECTED  | Fully connected layer of size 43, outputs 43x1|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Following setting was used for training the model:

* **Optimizer:** Adam Optimizer
* **Learning Rate:** 0.001
* **Loss Functiom:** Cross entropy
* **EPOCHS:** 50
* **keep probability(keep_prob):** When training: 0.7
* **BATCH SIZE:** 128
* **tf.truncated_normal():** mean = 0, and standard deviation = 0.1 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 



