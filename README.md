# Predict-ImageRotation-Angle

##  Requirement

please install the requirement.txt file before running the program

pip install -r requirements.txt


## Test the results

master_predict_rotation.py  is the program which accepts two images and predict the roatation of the second imaged formed by rotating the first.

python master_predict_rotation.py test/1.png test/1_ro_90.png 
python master_predict_rotation.py test/2.png test/2_ro_180.png

Input should be the path of the images.
Input image must be of 200 x 200 dimension


## Train using turi create


Put all your images in the data folder and delete data insdie the folder 0 , 1 , 2 , 3 subfolder of coreml_data

Run following program in sequence

python prepdare_data_for_coreML.py
python train_using_coreml.py

## Train using Tensorflow

Put training images into data folder.

python train_using_tensorflow.py
