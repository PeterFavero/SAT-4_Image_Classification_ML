import numpy as np
import joblib #used for joblib.dump and joblib.load
import matplotlib.pyplot as plt

#Have to do something wierd to import torch (ask about this)
import sys
sys.path.append('/Users/2.peterfaveroproductive/Library/Python/3.9/lib/python/site-packages')
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import optim



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
train_x = np.array( joblib.load('preprocessed/train_x_preprocessed') ) # TRAIN_SIZE x 6 
train_y = np.array( joblib.load('loaded/train_y_loaded') ) # TRAIN_SIZE x 4
test_x = np.array( joblib.load('preprocessed/test_x_preprocessed') ) # TEST_SIZE x 6 
test_y = np.array( joblib.load('loaded/test_y_loaded') ) # TRAIN_SIZE x 4
print(" -- Datasets loaded.\n")

#Make sure we haven't made any errors with DataPreprocessing.py
assert( len(train_x) == len(train_y) )
assert( len(train_x) == TRAIN_SIZE )
assert( len(test_x) == len(test_y) )
assert( len(test_x) == TEST_SIZE )
assert( len(train_x[0]) == len(test_x[0]) ) 

#Define the length of feature vectors
num_features = len(train_x[0])

#Declare the model; num_features nueron input -> 50 neuron hidden layer 
#                   -> 100 neuron hidden layer -> 4 neuron output layer
model = nn.Sequential(
    nn.Linear(num_features, 40),
    nn.ReLU(),
    nn.Linear(40, 50),
    nn.ReLU(),
    nn.Linear(50, 4)
)

#Declare loss, optimizer, and epochs, and batch size
loss_function = nn.CrossEntropyLoss() #Good for classification
optimizer = optim.Adam(model.parameters(), lr=0.0005) #Adam optimizer, apparently very popular
num_epochs = 300
batch_size = 200
terminated_early = False

print(' -- Beginning model training: max ' + str(num_epochs) + 
      ' epochs, batch_size = ' + str(batch_size) +'.')
for epoch in range(num_epochs) :

    for i in range(0, TRAIN_SIZE, batch_size) :
        
        #Get the set of predicted values from the model for each feature vector in the batch
        predicted_y = model(torch.FloatTensor(train_x[i:i+batch_size]))

        #Compute loss for this batch
        loss = loss_function(predicted_y, torch.FloatTensor(train_y[i:i+batch_size]))
        
        #Do gradient descent 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

    print('      * Epoch #' + str(epoch) + '\t| loss = ' + str(loss.item()) + '.')
    if( loss.item() < 0.025 ) :
        print(' -- Model training terminated by sufficiently low loss ( < 0.025 ):\n    max ' + 
              str(num_epochs) + ' epochs, batch_size = ' + str(batch_size) +'.')
        terminated_early = True
        break
    
if( not terminated_early ) : 
    print(' -- Model training terminated by epoch limit: ' + str(num_epochs) + 
      ' epochs, batch_size = ' + str(batch_size) +'.')

#Declare a variable to count how many test cases we correctly identify
correct = 0

#Test model accuracy
for i in range(TEST_SIZE) :
    if( test_y[i].argmax() == model(torch.FloatTensor(test_x[i])).argmax() ) :
        correct += 1
print(' -- Model tested, correctly identitifies ' + str(100*correct/TEST_SIZE) + '% of ' 
      + str(TEST_SIZE) + ' test cases.\n' )

joblib.dump(model, 'model/trained')
print(" -- Model dumped with joblib\n")

print("-------\nModelTraining.py terminated successfully.\n----\n")