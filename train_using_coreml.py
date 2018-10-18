
# coding: utf-8

# In[19]:


import turicreate as tc
import os


# In[10]:


# Load images (Note: you can ignore 'Not a JPEG file' errors)
data = tc.image_analysis.load_images('data_coreML', with_path=True)


# In[11]:


# From the path-name, create a label column
data['label'] = data['path'].apply(lambda path: os.path.dirname(path).split('/')[-1])


# In[12]:


# data.save('angle_rotation.sframe')


# In[13]:


# Explore interactively
#data.explore()


# In[14]:


# Load the data
#data =  tc.SFrame('angle_rotation.sframe')

# Make a train-test split
train_data, test_data = data.random_split(0.8)

# Create the model
model = tc.image_classifier.create(train_data, target='label' , max_iterations= 1000 , batch_size= 250 , model= "VisionFeaturePrint_Screen")


# Save predictions to an SArray
predictions = model.predict(test_data)

# Evaluate the model and save the results into a dictionary
metrics = model.evaluate(test_data)
print(metrics['accuracy'])


# In[16]:


# Save the model for later use in Turi Create
model.save('angleRoationPredict.model')

# Export for use in Core ML
#model.export_coreml('angleRoationPredict.mlmodel')




