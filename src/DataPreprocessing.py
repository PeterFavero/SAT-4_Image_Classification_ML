# TASK: Pre-processing
# You need to figure out how to pass in the images as feature vectors to the models.
# You should not simply pass in the entire image as a flattened vector;
# otherwise, it's very slow and just not really effective
# Instead you should extract relevant features from the images.
# Refer to Section 4.1 of https://arxiv.org/abs/1509.03602, especially first three sentences
# and consider what features you want to extract
# And like the previous task, once you have your pre-processed feature vectors,
# you may want to dump and load because pre-processing will also take a while each time.
# MAKE SURE TO PRE-PROCESS YOUR TEST SET AS WELL!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
import joblib #used for joblib.dump and joblib.load
import cv2

#Enhanced Vegetation Index
def EVI(red, blue, nir) :
    return 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)

#Normalized Vegtation Index
def NDVI(red, nir) :
    return (nir - red) / (nir + red)

#Atmospherically Resistant Vegetation Index
def ARVI(red, blue, nir) :
    return (nir - 2*red + blue) / (nir + 2*red + blue)

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

for i in range(100) :

    #create copies of current image in hsv format to access hue and sat
    train_image_hsv = cv2.cvtColor(train_x[i, :, :, 0:3], cv2.COLOR_RGB2HSV)
    test_image_hsv = cv2.cvtColor(test_x[i, :, :, 0:3], cv2.COLOR_RGB2HSV)

    """ unused training feautes
    #unused means (4)
    train_red_mean = np.mean(train_x[i, :, :, 0])
    train_green_mean = np.mean(train_x[i, :, :, 1])
    train_blue_mean = np.mean(train_x[i, :, :, 2])
    train_nir_mean = np.mean(train_x[i, :, :, 3])
    #unused standard deviations (4)
    train_red_std = np.std(train_x[i, :, :, 0])
    train_green_std = np.std(train_x[i, :, :, 1])
    train_blue_std = np.std(train_x[i, :, :, 2])
    train_nir_std = np.std(train_x[i, :, :, 3])
    #unused indexes (3)
    train_evi = EVI(train_red_mean, train_blue_mean, train_nir_mean)
    train_ndvi = NDVI(train_red_mean, train_nir_mean)
    train_arvi = ARVI(train_red_mean, train_blue_mean, train_nir_mean)
    #unused intensity (1)
    train_int = (train_red_mean + train_green_mean + train_blue_mean + train_nir_mean)
    #standardize means on a scale from 0 to 1 by multiplying each of them by 1/intensity
    train_red_mean /= train_int
    train_green_mean /= train_int
    train_blue_mean /= train_int
    train_nir_mean /= train_int
    """

    #compute training 6 features 
    train_hue_mean = np.mean(train_image_hsv[:, :, 0])
    train_sat_mean = np.mean(train_image_hsv[:, :, 1])
    train_val_mean = np.mean(train_image_hsv[:, :, 2])
    train_hue_std = np.std(train_image_hsv[:, :, 0])
    train_sat_std = np.std(train_image_hsv[:, :, 1])
    train_val_std = np.std(train_image_hsv[:, :, 2])

    #append an array containing each training feature 
    #to the preprocessed training dataset
    train_x_preprocessed.append([
        train_hue_mean,
        train_sat_mean,
        train_val_mean,
        train_hue_std,
        train_sat_std,
        train_val_std
    ])

    """ unused testing features:
    #unused means (4)
    test_red_mean = np.mean(test_x[i, :, :, 0])
    test_green_mean = np.mean(test_x[i, :, :, 1])
    test_blue_mean = np.mean(test_x[i, :, :, 2])
    test_nir_mean = np.mean(test_x[i, :, :, 3])
    #unused standard deviations (4)
    test_red_std = np.std(test_x[i, :, :, 0])
    test_green_std = np.std(test_x[i, :, :, 1])
    test_blue_std = np.std(test_x[i, :, :, 2])
    test_nir_std = np.std(test_x[i, :, :, 3])
    #unused indexes (3)
    test_evi = EVI(test_red_mean, test_blue_mean, test_nir_mean)
    test_ndvi = NDVI(test_red_mean, test_nir_mean)
    test_arvi = ARVI(test_red_mean, test_blue_mean, test_nir_mean)
    #unused intensity (1)
    test_int = (test_red_mean + test_green_mean + test_blue_mean + test_nir_mean)
    #standardize means on a scale from 0 to 1 by multiplying each of them by 1/intensity
    test_red_mean /= test_int
    test_green_mean /= test_int
    test_blue_mean /= test_int
    test_nir_mean /= test_int
    """

    #computing 6 testing features
    test_hue_mean = np.mean(test_image_hsv[:, :, 0])
    test_sat_mean = np.mean(test_image_hsv[:, :, 1])
    test_val_mean = np.mean(test_image_hsv[:, :, 2])
    test_hue_std = np.std(test_image_hsv[:, :, 0])
    test_sat_std = np.std(test_image_hsv[:, :, 1])
    test_val_std = np.std(test_image_hsv[:, :, 2])

    #append an array containing each testing feature 
    #to the preprocessed testing dataset
    test_x_preprocessed.append([
        test_hue_mean,
        test_sat_mean,
        test_val_mean,
        test_hue_std,
        test_sat_std,
        test_val_std
    ])

joblib.dump(train_x_preprocessed, 'preprocessed/train_x_preprocessed')
joblib.dump(test_x_preprocessed, 'preprocessed/test_x_preprocessed')

print("-------\nDataPreprocessing.py terminated successfully.\n----\n")

#notes:

#14
#mean
#standard deviation
#variance
#2nd moment
#direct cosine transforms
#correlation
#co-variance
#autocorrelation
#energy
#entropy
#homogeneity
#contrast
#maximum probability
#sum

#hue, saturation, intensity (I think this the same as value), 
#and NIR channels as well as those of the color co-occurrence matrices

#From previous file: "You’re welcome to use the colorsys module to convert from RGB to HSV"

#After this file is run, run ModelTraining.py

#Evan says use hsv instead of rgb, only about 6 features
#Defnitely serialize
#Accuracy hopefully over 80%