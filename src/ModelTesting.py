#Import nescessary libraries
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import torch

if( os.listdir('model') != ['trainedMLP', 'trainedSVM'] ) :
    #run through DataLoading, DataPreprocessing, and ModelTraining as 
    #nescessary if both models aren't properly saved at this point.
    import run

#Defining function to transform the test_y from a N by 4 matrix
#into a length N array populated with values 0, 1, 2, 3 corresponding to the label of the data
def labelTransform(arr):
    arr = arr.tolist()
    transformed = []
    for subarr in arr:
        transformed.append(subarr.index(1))
    return transformed

print("\n----\nModelTesting.py successfully compiled & run.\n-------\n")

#load test dataset
TEST_SIZE = joblib.load('loaded/num_testing_entries')
test_x = np.array( joblib.load('preprocessed/test_x_preprocessed') ) # TEST_SIZE x 6 
test_y = np.array( joblib.load('loaded/test_y_loaded') ) # TRAIN_SIZE x 4

#load both models
model_MLP = joblib.load('model/trainedMLP')
model_SVM = joblib.load('model/trainedSVM')

print(" -- Datasets and models loaded.\n")

#Declare a variable to count how many MLP test cases we correctly identify
correct_MLP = 0

#Test MLP model accuracy
for i in range(TEST_SIZE) :
    if( test_y[i].argmax() == model_MLP(torch.FloatTensor(test_x[i])).argmax() ) :
        correct_MLP += 1
print(' -- MLP Model tested, correctly identitifies ' + str(100*correct_MLP/TEST_SIZE) + '% of ' 
    + str(TEST_SIZE) + ' test cases.\n' )

#Convert testing y values into appropriate shape for sklearn SVM funcitons
transformed_test_y = labelTransform(test_y)

#Make SVM predictions
train_x = np.array( joblib.load('preprocessed/train_x_preprocessed') )
standardizer = StandardScaler()
standardizer.fit(train_x)
predictions = model_SVM.predict(standardizer.transform(test_x))

#Declare a variable to count how many test cases we correctly identify
correct_SVM = 0

#Test SVM model accuracy
for i in range(TEST_SIZE) :
    if(predictions[i] == transformed_test_y[i]) :
        correct_SVM += 1
print(' -- SVM Model tested, correctly identitifies ' + str(100*correct_SVM/TEST_SIZE) + '% of ' 
    + str(TEST_SIZE) + ' test cases.\n' )

print("-------\nModelTesting.py terminated successfully.\n----\n")
