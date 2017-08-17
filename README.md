# **Traffic Sign Recognition** 

This project is created to recognize traffic signs through machine learning principles, with focus on deep learning and Convolutional neural networks. I used the pickled database derived from [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) for training, validating and testing the model. I have used few images downloaded from internet to test the model for accuracy.

---

## Building a Traffic Sign Recognition system ##

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: "./new-images/11 right of way.png"  "Sign1"
[image2]: ./new-images/12-priority-road.png  "Sign2"
[image3]: ./new-images/13-yield.png  "Sign3"
[image4]: ./new-images/17-no-entry.png  "Sign4"
[image5]: ./new-images/26-signal-lights.png  "Sign5"
[image6]: ./new-images/36-straight-or-right.png  "Sign6"
[image7]: ./new-images/38-keep-right.png  "Sign7"

## Rubric Points
In the text below, I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is the README file. My [project code](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb) is in the repo, along with a html file generated from the jupyter notebook.

### Data Set Summary & Exploration

1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3 
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

When the histogram of number of occurrances of each label is plotted, it may be observed that some of the labels are more (some are about 2000 while some are about 200) than the others. This could result in more training for few labels and less training for few labels, and may result in less accurate detection of some signs than the other. I decided to proceed with the dataset as such, and comeback to modify if I am not able to achieve the required accuracy.

![image](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/new-images/barchart.png)


### Designing and Testing a Model Architecture

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The implemented architecture is a slightly modified version of LeNeT-5. This included an additional fully-connected layer compared to LeNeT-5 and also included dropouts. 

The network has the following architecture:

* Layer 1: Convolution Layer to accept 32X32X1 images to convert them to 28X28X10 through a 5X5 filter and a stride of 1
* ReLu Activation
* MaxPool with a stride of 2 and 2X2 as window size
* Layer 2: Convolution Layer to accept 14X14X10 and to output 10X10X20 
* ReLu Activation
* Maxpool with a stride of 2 and 2X2 as window size
* Flatten the layer to result in 500 outputs
* Layer 3: Fully connected layer with 500 inputs, 700 outputs
* ReLu Activation
* Layer 4: Fully connected layer with 700 inputs and 200 outputs
* ReLu Activation
* Dropout with 0.65 keep fraction
* Layer 5: Fully connected layer with 200 inputs and 80 outputs
* ReLu activation
* Dropout with 0.65 keep fraction
* Layer 6: fully connected layer with 80 inputs and 43 outputs

With 40 epochs and a batch size of 100 images of grayscaled and normalized images, I was able to achieve an accuracy of 96.1% on the validation set and 93.8% on the test set. 




#### What didn't work for me ####

I attempted with the LeNeT-5 architecture for 3 channel images initially without pre-processing the images, at all. This resulted in an accuracy of about 70% on validation set. 

Then I went on to normalize the image with (pixelvalue-128)/128. This improved by accuracy to about 80%.

I further went ahead and normalized the accuracy of each color channel and reduced the standard deviation to 1 by subtracting the mean and dividing by the standard deviations calculated from numpy ( calculated as np.mean(), and np.std() ). This did not improve my validation accuracy, and it still settled at about 80%.

Then I thought that may be a fixed learning rate could be causing the problem for 

Then I went on to convert the images to gray scale and then apply the normalization ( subtract by the mean and divide by the standard deviation). This gave me an validation accuracy of close to 90%. 

Then I reduced the 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

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
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


