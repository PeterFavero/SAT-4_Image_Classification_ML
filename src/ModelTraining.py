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
train_x = joblib.load('preprocessed/train_x_preprocessed')
train_y = joblib.load('loaded/train_y_loaded')
test_x = joblib.load('preprocessed/test_x_preprocessed')
test_y = joblib.load('loaded/test_y_loaded')

#Creating training dataframe
index_values = np.arange(len(train_x))
column_values = ["Hue Mean", "Sat Mean", "Val Mean", "Hue STD", "Sat STD", "Val STD"]

train_df = pd.DataFrame(data = train_x, index = index_values, columns = column_values)

#Creating testing dataframe
index_values = np.arange(len(test_x))
test_df = pd.DataFrame(data = test_x, index = index_values, columns = column_values)

print("-------\nModelTraining.py terminated successfully.\n----\n")