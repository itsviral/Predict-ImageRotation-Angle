
# coding: utf-8

# In[22]:


import turicreate as tc
import sys
import cv2
import numpy as np


# In[21]:


image_size = 200 , 200

if (len(sys.argv) == 3):
    
    try:
        image1 = cv2.imread(sys.argv[1])
        image2 = cv2.imread(sys.argv[2])
            
    except:
        
            print("problem while opening image")
            
    # Read image in grayscale and resize
            
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
    image1 = cv2.resize(image1 , image_size)  
    image2 = cv2.resize(image2 , image_size)  

    concated_image = np.concatenate((image1, image2), axis=0)

    cv2.imwrite("temp.png" , concated_image)

    try:
        
        model = tc.load_model("angleRoationPredict.model")

        tc_image = tc.Image("temp.png")

        result = model.predict(tc_image)
       
        ref = 1
        
        if (result == "0"):
            
            ref = 0
            
        if (result == "1"):
            
            ref = 90
            
        if (result == "2"):
            
            ref = 180
            
        if (result == "3"):
            
            ref = 270
            
        
        print("The angle of rotation is : " +  str(ref))
        
    except:
        
         print("problem while predicting image roatation")

