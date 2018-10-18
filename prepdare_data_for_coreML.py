
# coding: utf-8

# In[4]:

from __future__ import division, print_function, absolute_import
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
import random
import uuid


# In[5]:


#..... All varialbe and parameters

# image size to be shrinked
image_size = 200 , 200

# assign the percentage of test data 
test_split_data_percentage = 20


# In[6]:


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
                
                fname = "data_coreML/" + str(temp) + "/" + uuid.uuid4().hex + ".png"
                
                cv2.imwrite(fname , concated_image)
        
        except:
            
            print("some error occured while opening file")

