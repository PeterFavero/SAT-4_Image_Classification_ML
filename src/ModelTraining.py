import numpy as np
import joblib #used for joblib.dump and joblib.load
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#Have to do something wierd to import torch (ask about this)
import sys
sys.path.append('/Users/2.peterfaveroproductive/Library/Python/3.9/lib/python/site-packages')
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import optim
torch.manual_seed(10)

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

#Defining a function for MLP.
#If you run this as is with torch.manual_seed(10) as above and the parameters listed below, you 
#should get a model with 99.711% accuracy in 2403 epoches, about an hour of training
#on my machine (Macbook air w/ M2 chip)
def MLP():
    print(' -- Running MLP model: TRAIN_SIZE = ' + str(TRAIN_SIZE) + ' TEST_SIZE = ' + str(TEST_SIZE) + '.\n')
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
    optimizer = optim.Adam(model.parameters(), lr=0.00025) #Adam optimizer, apparently very popular
    num_epochs = 10000
    batch_size = 200
    terminated_early = False

    #For visualization
    epoches = []
    losses = []

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

        epoches.append(epoch)
        losses.append(loss.item())

        print('      * Epoch #' + str(epoch) + '\t| loss = ' + str(loss.item()) + '.')
        if( loss.item() < 0.001 ) :
            print(' -- Model training terminated by sufficiently low loss ( < 0.001 ):\n    max ' + 
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

    joblib.dump(model, 'model/trainedMLP')
    print(" -- MLP Model dumped with joblib\n")

    plt.plot(epoches, losses)
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoches")

    plt.savefig("visualizations/MLP_visualization.png")

#Run the MLP model
MLP()

#Define a function for SVM
def SVM(c_value):
    print(' -- Running SVM model: TRAIN_SIZE = ' + str(TRAIN_SIZE) + ' TEST_SIZE = ' + str(TEST_SIZE) + '.\n')
    #Defining function to transform the test_y from a N by 4 matrix
    #into a length N array populated with values 0, 1, 2, 3 corresponding to the label of the data
    def labelTransform(arr):
        arr = arr.tolist()
        transformed = []
        for subarr in arr:
            transformed.append(subarr.index(1))

        return transformed
    
    #Convert training and testing y values into appropriate shape for sklearn SVM funcitons
    transformed_train_y = labelTransform(train_y)
    transformed_test_y = labelTransform(test_y)

    #Standardize training and testing x values
    standardizer = StandardScaler()

    #Fit and standardize train_x at the same time
    standardized_train_x = standardizer.fit_transform(train_x)

    #Standardize test_x
    standardized_test_x = standardizer.transform(test_x)

    #Create SVM model
    model = SVC(decision_function_shape="ovo", kernel="rbf", C=c_value)

    print(' -- Training model\n')

    #Train model
    model.fit(standardized_train_x, transformed_train_y)

    #Make predictions
    predictions = model.predict(standardized_test_x)

    #Test accuracy
    
    #Declare a variable to count how many test cases we correctly identify
    correct = 0

    #Test model accuracy
    for i in range(TEST_SIZE) :
        if(predictions[i] == transformed_test_y[i]) :
            correct += 1
    print(' -- Model tested, correctly identitifies ' + str(100*correct/TEST_SIZE) + '% of ' 
        + str(TEST_SIZE) + ' test cases.\n' )
    
    #Only dump on the most accurate c value
    if(c_value == 100000):
        joblib.dump(model, 'model/trainedSVM')
        print(" -- SVM Model dumped with joblib\n")

    return 100*correct/TEST_SIZE

#Create visualization of SVM using C values of 0.1, 1, 10, 100, 1000, 10000, 100000
c_values = []
percentages = []

for c_multiplier in range(-1, 6):
    #Set value of c to 10^c_multiplier
    c_value = 10 ** c_multiplier

    print(f'----- Testing C value of {c_value} -----\n')

    #Run the SVM model
    percentage = SVM(c_value)

    c_values.append(c_value)
    percentages.append(percentage)

fig, ax = plt.subplots(layout="constrained")

ax.plot(c_values, percentages, marker=".", markersize=10)

ax.set_xlabel("C Value (Log Base 10 Scale)")
ax.set_ylabel("Percentage Correct (%)")
ax.set_title("C Value vs. Percentage Correct (%)")

ax.set_xscale("log", base=10)

plt.savefig("visualizations/SVM_Visualization.png")

print("--- Saved Figure ---\n")

print("-------\nModelTraining.py terminated successfully.\n----\n")