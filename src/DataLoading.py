'''
HW: Satellite Image Classification
You may work on this assignment with ONE partner.

We use the SAT-4 Airborne Dataset: https://www.kaggle.com/datasets/crawford/deepsat-sat4.
Download "sat-4-full.mat" from Kaggle and place it in your working dir.

This dataset is large (500K satellite imgs of the Earth's surface).
Each img is 28x28 with 4 channels: red, green, blue, and NIR (near-infrared).
The imgs are labeled to the following 4 classes: 
barren land | trees | grassland | none

The MAT file from Kaggle contains 5 variables:
- annotations (explore this if you want to)
- train_x (400K training images), dim: (28, 28, 4, 400000)
- train_y (400k training labels), dim: (4, 400000)
- test_x (100K test images), dim: (28, 28, 4, 100000)
- test_y (100K test labels), dim: (4, 100000)

For inputs (train_x and test_x):
0th and 1st dim encode the row and column of pixel.
2nd dim describes the channel (RGB and NIR where R = 0, G = 1, B = 2, NIR = 3).
3rd dim encodes the index of the image.

Labels (train_y and test_y) are "one-hot encoded" (look this up).

Your task is to develop two classifiers, SVMs and MLPs, as accurate as you can.
'''

#Run this first

#Import nescessary libraries
import numpy as np
import scipy
import joblib #used for joblib.dump and joblib.load

print("\n----\nDataLoading.py successfully compiled & run.\n-------\n")

#Constant values indicating the size of our original datasets (do not change)
MAT_TRAIN_SIZE = 400000
MAT_TEST_SIZE = 100000

#Load in the dataset using scipy.io.loadmat
mat_data = scipy.io.loadmat("archive/sat-4-full.mat")
print(" -- Archive loaded.\n")

#Declare size of training and testing subsets
#Dealing with 400K and 100K images will take forever, so do
#training and testing on small subsets (Andrew did 10K and 2.5K, tune as you need).
num_training_entries = 10000
num_testing_entries = 2500
joblib.dump(num_training_entries, 'loaded/num_training_entries')
joblib.dump(num_testing_entries, 'loaded/num_testing_entries')

#Randomly selected all indices used for training and testing 
selected_training_indices = np.random.choice(MAT_TRAIN_SIZE, num_training_entries, replace=False)
print(" --", num_training_entries, "random training indices selected.")
selected_testing_indices = np.random.choice(MAT_TEST_SIZE, num_testing_entries, replace=False)
print(" --", num_testing_entries, "random testing indices selected.\n")

#Declare training and testing datsets, and store the associated values into them
train_x = []
train_y = [] 
test_x = []
test_y = []

print(" -- Beginning training dataset storage:")
for i in range(num_training_entries) :
    train_x.append(mat_data['train_x'][:, :, :, selected_training_indices[i]])
    train_y.append(mat_data['train_y'][:, selected_training_indices[i]])
    if( i % 100 == 0 ) : print("      * Stored point #" + str(i))
print(" -- Training dataset (" + str(num_training_entries) + " entries) stored.")
#Save test set using dump so we don't have to load in the full dataset in subsequent runs
joblib.dump(train_x, 'loaded/train_x_loaded')
joblib.dump(train_y, 'loaded/train_y_loaded')
print(" -- Training dataset dumped with joblib.\n")

print(" -- Beginning testing dataset storage:")
for i in range(num_testing_entries) :
   test_x.append(mat_data['test_x'][:, :, :, selected_testing_indices[i]])
   test_y.append(mat_data['test_y'][:, selected_testing_indices[i]])
   if( i % 100 == 0 ) : print("      * Stored point #" + str(i))
print(" -- Testing dataset (" + str(num_testing_entries) + " entries) stored.")
#Save test set using dump so we don't have to load in the full dataset in subsequent runs
joblib.dump(test_x, 'loaded/test_x_loaded')
joblib.dump(test_y, 'loaded/test_y_loaded')
print(" -- Testing dataset dumped with joblib.\n")

print("-------\nDataLoading.py terminated successfully.\n----\n")

#After this file is run, run DataPreprocessing.py