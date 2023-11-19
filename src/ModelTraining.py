import numpy as np
import pandas as pd
import joblib #used for joblib.dump and joblib.load

# TASK: TRAIN YOUR MODEL
# You have your feature vectors now, time to train.
# Again, train two models: SVM and MLP.
# Make them as accurate as possible. Tune your hyperparameters.
# Check for overfitting and other potential flaws as well.

print("\n----\nModelTraining.py successfully compiled & run.\n-------\n")

#Load size of training and testing subsets (constant here, do not edit)
TRAIN_SIZE = joblib.load('loaded/num_training_entries')
TEST_SIZE = joblib.load('loaded/num_testing_entries')

#Load training and testing arrays
train_x = joblib.load('preprocessed/train_x_preprocessed') # TRAIN_SIZE x 6 
train_y = joblib.load('loaded/train_y_loaded') # TRAIN_SIZE x 4
test_x = joblib.load('preprocessed/test_x_preprocessed') # TEST_SIZE x 6 
test_y = joblib.load('loaded/test_y_loaded') # TRAIN_SIZE x 4

#Declare labels
column_values = ["Hue Mean", "Sat Mean", "Val Mean", "Hue STD", "Sat STD", "Val STD"]

#Creating training dataframe
index_values = np.arange(len(train_x))
train_df = pd.DataFrame(data = train_x, index = index_values, columns = column_values)

#Create array of training output labels and add it to the training dataframe
train_y_labels = []
for i in range(TRAIN_SIZE) :
    train_y_labels.append( 'Barren' if train_y[i][0] == 1 else 'Trees' if train_y[i][1] == 1 
                          else 'Grassland' if train_y[i][2] == 1 else 'None')
train_df['Label'] = train_y_labels

#Creating testing dataframe
index_values = np.arange(len(test_x))
test_df = pd.DataFrame(data = test_x, index = index_values, columns = column_values)

#Create array of testing output labels
test_y_labels = []
for i in range(TEST_SIZE) :
    test_y_labels.append( 'Barren' if test_y[i][0] == 1 else 'Trees' if test_y[i][1] == 1 
                          else 'Grassland' if test_y[i][2] == 1 else 'None')
test_df['Label'] = test_y_labels

print(train_df)
print()
print(test_df)


print("-------\nModelTraining.py terminated successfully.\n----\n")