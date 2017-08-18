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




## Rubric Points

In the text below, I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. 

---

### Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. 

The submission includes the project code and this is the README file. My [project code](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb) is in the repo, along with a html file generated from the jupyter notebook.

### Data Set Summary & Exploration

1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the trafficsigns data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3 
* The number of unique classes/labels in the data set is 43

2. Include an exploratory visualization of the dataset.

When the histogram of number of occurrances of each label is plotted, it may be observed that some of the labels are more (some are about 2000 while some are about 200) than the others. This could result in more training for few labels and less training for few labels, and may result in less accurate detection of some signs than the other. I decided to proceed with the dataset as such, and comeback to modify if I am not able to achieve the required accuracy.

![image](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/new-images/barchart.png)

### Designing and Testing a Model Architecture

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The implemented architecture is a slightly modified version of LeNeT-5. This included an additional fully-connected layer compared to LeNeT-5 and also included dropouts. 

#### pre-processing ####

I converted the images to grayscale so that the memory required to process the network is smaller. Additionally, to make the average value of the pixel values zero, subtracted mean of each image pizel value (np.mean()) from the image, and then to reduce the scale of the values, divided the pixel value with standard deviation (np.std()). This resulted in most of the values for the pixels represented in a narrow band(less than 1), and they are all centered about 0.


2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

* Convolution Layer to accept 32X32X1 images to convert them to 28X28X10 through a 5X5 filter and a stride of 1
* ReLu Activation
* MaxPool with a stride of 2 and 2X2 as window size
* Convolution Layer to accept 14X14X10 and to output 10X10X20 
* ReLu Activation
* Maxpool with a stride of 2 and 2X2 as window size
* Flatten the layer to result in 500 outputs
* Fully connected layer with 500 inputs, 700 outputs
* ReLu Activation
* Fully connected layer with 700 inputs and 200 outputs
* ReLu Activation
* Dropout with 0.85 keep fraction
* Fully connected layer with 200 inputs and 80 outputs
* ReLu activation
* Dropout with 0.85 keep fraction
* Fully connected layer with 80 inputs and 43 outputs

I used a learning rate of 0.001, and an epoch count of 40 and a batch size of 400.

3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam Optimizer. This choice is partly because of my lack of awareness about intricate usage details of these optimizers. I went ahead with the suggested optimizer from the class tutorials as mentioned in LeNet lab. 

I had initially tried to use a low batch size and low epoch count for tuning the model. However, this resulted in the low validation accuracies. Then I increased the epochs to over 30 and also incresed the batch size. Although this increases the system memory requirements, I was optimistic about better fitting model.

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I attempted with the LeNeT-5 architecture for 3 channel images initially without pre-processing the images, at all. This resulted in an accuracy of about 70% on validation set. The LeNet was the initial choice because it was available from the class quiz. 

Then I went on to normalize the image with (pixelvalue-128)/128. This improved by accuracy to about 80%.

I further went ahead and normalized the accuracy of each color channel and reduced the standard deviation to 1 by subtracting the mean and dividing by the standard deviations calculated from numpy ( calculated as np.mean(), and np.std() ). This did not improve my validation accuracy, and it still settled at about 80%.

Then I went on to convert the images to gray scale and then apply the normalization ( subtract by the mean and divide by the standard deviation). This gave me an validation accuracy of close to 90%. 

I included dropout layers to avoid overfitting the model. 

Then I reduced the learning rate and tried to change the batch sizes to achieve reasonagle validation accuracy.

I used the number of epochs, batchsize, and the learning rate as my hyperparameters for achieving a validation accuracy of over 93%. 

Some of the important design choices:

* increasing the depth of convolution layers - to be able to extract and distinguish more features
* including dropout layer - to reduce over fitting the model
* including another fully connected layer - to increase the possibility of isolating features

My final model results were:

* training set accuracy of   1.000
* validation set accuracy of 0.949
* test set accuracy of       0.930

### Test a Model on New Images

1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![Right of way](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/new-images/11%20right%20of%20way.png)
![Priority road](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/new-images/12%20priority%20road.png)
![yield](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/new-images/13%20yield.png)
![no entry](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/new-images/17%20no%20entry.png)
![signal lights](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/new-images/26%20signal%20lights.png)
![Straight or right](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/new-images/36%20straight%20or%20right.png)
![keep right](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/new-images/38%20keep%20right.png)

The image for traffic signs would be difficult to classify. That is because, when this image is convertd to grayscale, this appears closer to caution sign. The images for stay right and no-entry do not have full circle. This could result in some of the features not being detected.

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image           |     Prediction              | 
|:---------------------:|:---------------------------------------------:| 
| Right of way       | Right of way          | 
| Priority road   | Priority road         |
| Yield     | Yield           |
| no-entry         | no-entry              |
| Traffic signs   | Caution                   |
| Straight of right     | Straight of right                             |
| Keep right            | Keep right                                    |

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.7%. This is less than the accuracy on the test set of 93%. The accuracy of new image classification changes drastically even with one prediction going wrong. That being said, the wrong classification of the new images may be partly because of imprefections included in the new images. 

The image below shows the color image considered, and the grayscale version of the image followed by the top three predictions.

![predictions](https://github.com/saras152/Traffic-Sign-Classifier/blob/master/new-images/predictions.png)

3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the image with traffic signs, the model predicted it to be caution. But the model's prediction is limited to 58%. But, the top three predictions don't even have the traffic signs. 

| Probability          |     Prediction              | 
|:---------------------:|:---------------------------------------------:| 
| 0.999999881       | Right of way          | 
| 0.999999285   | Priority road         |
| 1.000000000   | Yield           |
| 0.999949694      | no-entry              |
| 0.581185758   | Caution                   |
| 0.999098659           | Straight or right                             |
| 0.985288680           | Keep right                                    |


## EDIT ##
I attempted a RBG image input based model instead of grayscale images as inputs to the network with the hopes that the traffic signs image will be corectly classified, but it was not. May be there is something else that needs to be done to correctly classify that image.

## What else can be done ##

The following things may be tried to improve the accuracy

* Color images of three channels may be used as they are for providing the inputs to the network.
    * This improves the classification accuracy by looking for colors too.
* Both the grayscale and color images may be provided to the network with a layer combining these two inputs
    * this provides an opportunity to the network to look at several possiblt characteristics
* The labels may need to be multi-dimentions. i.e. we may include their type - warning signs, hazard signs, information signs, etc. This improves the network's ability for predictions when it encounters any signs that were not in the training set. 
* The network needs to be trained to assess the sub-classification
    * it should be able to identify 'speed limit of 30 ends' from its previous observations on 'speed limit is 30' and 'speed limit of 80 ends here'. My present model is not capable of such improvements. 
* A global optimization scheme has to be incorporated which will also consider changes to learning rate, epochs, batchsize and any such hyperparameters that would result in automated tuning of the model.
    * the optimization scheme should have control on the parameters' range and their step size to shit our situational requirements. 
