import numpy as np
import joblib #used for joblib.dump and joblib.load
import cv2

#TASK: Pre-Processing x datasets into feature vectors
#Sources some useful indices from this paper:
#https://arxiv.org/abs/1509.03602

#Feature calculation methods
#Enhanced Vegetation Index
def EVI(red, blue, nir) :
    return 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
#Normalized Vegtation Index
def NDVI(red, nir) :
    return (nir - red) / (nir + red)
#Atmospherically Resistant Vegetation Index
def ARVI(red, blue, nir) :
    return (nir - 2*red + blue) / (nir + 2*red + blue)

#Feature vector appending method
def append_features(preprocessed, image_hsv, image_rgbn) :
     
     #means used to compute later features
     red_mean = np.mean(image_rgbn[:, :, 0])
     blue_mean = np.mean(image_rgbn[:, :, 2])
     
     #computing 6 testing features
     hue_mean = np.mean(image_hsv[:, :, 0])
     sat_mean = np.mean(image_hsv[:, :, 1])
     val_mean = np.mean(image_hsv[:, :, 2])
     hue_std = np.std(image_hsv[:, :, 0])
     sat_std = np.std(image_hsv[:, :, 1])
     val_std = np.std(image_hsv[:, :, 2])

     nir_mean = np.mean(image_rgbn[:, :, 3])
     nir_std = np.std(image_rgbn[:, :, 3])

     evi = EVI(red_mean, blue_mean, nir_mean)
     ndvi = NDVI(red_mean, nir_mean)
     arvi = ARVI(red_mean, blue_mean, nir_mean)

     #append an array containing each testing feature 
     #to the preprocessed dataset
     preprocessed.append([
         hue_mean,
         sat_mean,
         val_mean,
         nir_mean,
         hue_std,
         sat_std,
         val_std,
         nir_std,
         evi, 
         ndvi,
         arvi, 
     ])

print("\n----\nDataPreprocessing.py successfully compiled & run.\n-------\n")

#Load size of training and testing subsets (constant here, do not edit)
TRAIN_SIZE = joblib.load('loaded/num_training_entries')
TEST_SIZE = joblib.load('loaded/num_testing_entries')

#Load datasets from DataLoading.py
train_x = np.array(joblib.load('loaded/train_x_loaded')) #TRAIN_SIZE x 28 x 28 x 4
train_y = np.array(joblib.load('loaded/train_y_loaded')) #TRAIN_SIZE x 4
test_x = np.array(joblib.load('loaded/test_x_loaded')) #TEST_SIZE x 28 x 28 x 4
test_y = np.array(joblib.load('loaded/test_y_loaded')) #TEST_SIZE x 4
print(" -- Datasets loaded.\n")

#Declare empty preprocessed datasets
train_x_preprocessed = []
test_x_preprocessed = []

print(" -- Beginning training dataset preprocessing:")
for i in range(TRAIN_SIZE) :

    #create copies of current image in hsv format to access hue and sat
    train_image_hsv = cv2.cvtColor(train_x[i, :, :, 0:3], cv2.COLOR_RGB2HSV)

    append_features(train_x_preprocessed, train_image_hsv, train_x[i])

    #print to keep track of progress
    if( i % 1000 == 0 ) : print("      * Preprocessed and stored point #" + str(i))
    
print(" -- Training dataset (" + str(TRAIN_SIZE) + " entries) preprocessed and stored.")
#Save test set using dump so we don't have to load in the full dataset in subsequent runs
joblib.dump(train_x_preprocessed, 'preprocessed/train_x_preprocessed')
print(" -- Training dataset dumped with joblib.\n")

print(" -- Beginning testing dataset preprocessing:")
for i in range(TEST_SIZE) :

    #create copies of current image in hsv format to access hue and sat
    test_image_hsv = cv2.cvtColor(test_x[i, :, :, 0:3], cv2.COLOR_RGB2HSV)

    append_features(test_x_preprocessed, test_image_hsv, test_x[i])

    #print to keep track of progress
    if( i % 1000 == 0 ) : print("      * Preprocessed and stored point #" + str(i))

print(" -- Testing dataset (" + str(TEST_SIZE) + " entries) preprocessed and stored.")
#Save test set using dump so we don't have to load in the full dataset in subsequent runs
joblib.dump(test_x_preprocessed, 'preprocessed/test_x_preprocessed')
print(" -- Testing dataset dumped with joblib.\n")

print("-------\nDataPreprocessing.py terminated successfully.\n----\n")