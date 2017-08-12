# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 13:39:12 2017

@author: raghunath
"""

# Load pickled data
import pickle
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
#%matplotlib inline

def plotmyimage(img,title=''):
    #This function is used during the debug phase. 
    #This just plots the image 
    plt.figure(figsize=(1,1))
    plt.imshow(img,cmap='gray') 
    plt.title(title)
    plt.show()

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    weight1=tf.Variable(tf.truncated_normal(shape=(5,5,3,16), mean = mu, stddev = sigma))
    bias1=tf.Variable(tf.zeros(16))
    conv_layer=tf.nn.conv2d(x,weight1,strides=[1,1,1,1],padding='VALID')+ bias1
    # TODO: Activation.
    conv_layer=tf.nn.relu(conv_layer)
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv_layer=tf.nn.max_pool(conv_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    weight2=tf.Variable(tf.truncated_normal(shape=(5,5,16,26), mean = mu, stddev = sigma))
    bias2=tf.Variable(tf.zeros(26))
    conv_layer2=tf.nn.conv2d(conv_layer,weight2,strides=[1,1,1,1],padding='VALID')+ bias2
    # TODO: Activation.
    conv_layer2=tf.nn.relu(conv_layer2)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv_layer2=tf.nn.max_pool(conv_layer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc0=flatten(conv_layer2)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(650, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    # TODO: Activation.
    fc1    = tf.nn.relu(fc1)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    # TODO: Activation.
    fc2    = tf.nn.relu(fc2)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



training_file = './traffic-signs-data/train.p'
validation_file='./traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']



n_train = X_train.shape[0]
n_validation = X_valid.shape[0]
n_test =X_test.shape[0]
image_shape = X_test.shape[1:]
n_classes = np.unique(y_train).shape[0]
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



randindex=random.randint(0,len(X_train))
print('A sample image shown below is at index {} with a label {}:'.format(randindex,y_train[randindex]))
randimage=X_train[randindex]
plotmyimage(randimage)

X_train=(X_train.astype(np.int16)-128)/128
X_valid=(X_valid.astype(np.int16)-128)/128
X_test=(X_test.astype(np.int16)-128)/128

imageH,imageW,imageD=image_shape


x = tf.placeholder(tf.float32,shape=[None,imageH,imageW,imageD])
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)



rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()




X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 10
BATCH_SIZE=128
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
print('Execution done. ')