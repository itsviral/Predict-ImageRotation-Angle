
# coding: utf-8

# In[1]:

from __future__ import division, print_function, absolute_import
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
import random



# In[2]:


#..... All varialbe and parameters

# image size to be shrinked
image_size = 50 , 50
image_data_and_lables = []

# assign the percentage of test data 
test_split_data_percentage = 20


# In[3]:


# ...... Make data

directory_path = "data"

file_counter = 1 

image_rotation_main_counter = 0

for root, dirs, filenames in os.walk(directory_path):
    
    total_files = len(filenames)
  
    for fileName in filenames:
        
         #Print Process Status
            
        if (file_counter % 100 == 0):
            
            percentage = (file_counter * 100) / total_files 
            
            print("image reading and processing status :  " + str(percentage) + "%")
            
        file_counter = file_counter + 1
        
        if(file_counter == total_files):
            
             print("image reading and processing status : 100%")
                
                
         # file io operation
                        
        image_file_path_name = (directory_path + "/" + fileName) 
        
        try:
            
            image = cv2.imread(image_file_path_name)
            
         # Read image in grayscale and resize
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            image = cv2.resize(image , image_size)  
          
            rot = [0 , 90 , 180, 270]
        
            for i in rot:
                
                rows,cols = image.shape
                
                temp = []
                
                if(i == 0):
                    
                    temp = 0
                    
                if(i == 90):
                    temp = 1
                    
                if(i == 180):
                    
                    temp = 2
                    
                if(i == 270):
                    
                    temp = 3

                M = cv2.getRotationMatrix2D((cols/2,rows/2), i ,1)
                
                new_roatated_image = cv2.warpAffine(image,M,(cols,rows))
                
                concated_image = np.concatenate((image, new_roatated_image), axis=0)
                
                image_data_and_lables.append((np.array(concated_image , dtype = "float32")/255.0 , temp))
        
        except:
            
            print("some error occured while opening file")
      


# In[4]:


# prepare data set for training

random.shuffle(image_data_and_lables)

total_data_len = len(image_data_and_lables)

split_len = int((total_data_len * test_split_data_percentage) / 100 )

x_train = []
y_train = []
x_test = []
y_test = []

split_counter = 0

for input_data, out_label in image_data_and_lables:
    
    if (split_counter > split_len):
        
        x_train.append(input_data)
        y_train.append(out_label)
        
    else:
    
        x_test.append(input_data)
        y_test.append(out_label)
        
    split_counter = split_counter + 1
    
del image_data_and_lables
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print("Shape of input image" , x_train.shape)


# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(800, activation=tf.nn.relu),
#   tf.keras.layers.Dense(800, activation=tf.nn.sigmoid),
#   tf.keras.layers.Dense(800, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(299, activation=tf.nn.softmax)
# ])
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# 
# model.fit(x_train, y_train, epochs= 1)
# model.evaluate(x_test, y_test)

# In[5]:


# Training Parameters
learning_rate = 0.001
num_steps = 111
batch_size = 128

# Network Parameters
num_input = 5000 
num_classes = 4
dropout = 0.25 

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
   
    with tf.variable_scope('ConvNet', reuse=reuse):
      
        x = x_dict['images']

        x = tf.reshape(x, shape=[-1, 100, 50, 1])

        
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

       
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer
        fc1 = tf.layers.dense(fc1, 2048)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': x_train}, y= y_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': x_test}, y= y_test,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])

