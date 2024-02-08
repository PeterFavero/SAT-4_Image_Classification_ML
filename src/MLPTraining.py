import numpy as np
import joblib #used for joblib.dump and joblib.load
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch import optim
torch.manual_seed(10)

#TASK: TRAIN MODELS

print("\n----\nMLPTraining.py successfully compiled & run.\n-------\n")

#Defining a function that trains the MLP model.
#If you run this as is with torch.manual_seed(10) as above and the parameters listed below, you 
#should get a model with 99.711% accuracy in 2403 epoches.
def MLP():

    #Define device to perform training on (testing integrating this mps right now in my own branch, currently works best on Cuda, MPS is slower than CPU)
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    #Load size of training and testing subsets (constant here, do not edit)
    TRAIN_SIZE = joblib.load('loaded/num_training_entries')
    TEST_SIZE = joblib.load('loaded/num_testing_entries')

    #Load tensors 
    train_x = torch.FloatTensor( np.array( joblib.load('preprocessed/train_x_preprocessed') ) ).to(device) # TRAIN_SIZE x num_features 
    print(" -- train_x loaded.")
    train_y =  torch.FloatTensor( np.array( joblib.load('loaded/train_y_loaded') ) ).to(device) # TRAIN_SIZE x 4
    print(" -- train_y loaded.")
    test_x =  torch.FloatTensor( np.array( joblib.load('preprocessed/test_x_preprocessed') ) ) # TEST_SIZE x num_features  
    print(" -- test_x loaded.")
    test_y =  torch.FloatTensor( np.array( joblib.load('loaded/test_y_loaded') ) ) # TRAIN_SIZE x 4
    print(" -- test_y loaded.")
    print(" -- All datasets loaded.\n")

    #Make sure we haven't made any errors with DataPreprocessing.py
    assert( len(train_x) == len(train_y) )
    assert( len(train_x) == TRAIN_SIZE )
    assert( len(test_x) == len(test_y) )
    assert( len(test_x) == TEST_SIZE )
    assert( len(train_x[0]) == len(test_x[0]) ) 

    #Define the length of feature vectors
    num_features = len(train_x[0])

    print(' -- Training MLP model: TRAIN_SIZE = ' + str(TRAIN_SIZE) + ', TEST_SIZE = ' + str(TEST_SIZE) + ', device = ' + device_string + '.\n')

    #Declare the model; num_features nueron input -> 40 neuron hidden layer 
    #                   -> 50 neuron hidden layer -> 4 neuron output layer
    model = nn.Sequential(
        nn.Linear(num_features, 40),
        nn.ReLU(),
        nn.Linear(40, 50),
        nn.ReLU(),
        nn.Linear(50, 4)
    ).to(device)

    #Declare loss, optimizer, and epochs, and batch size
    
    #Good for classification 
    loss_function = nn.CrossEntropyLoss() #Automatically applies softmax, do not apply it yourself 
    optimizer = optim.Adam(model.parameters(), lr=0.00025) 
    num_epochs = 10000
    batch_size = 200
    loss_threshold = 0.01
    terminated_early = False

    #For visualization
    epoches = []
    losses = []

    start_time = time.time()
    prev_time = start_time
    print(' -- Beginning model training: max ' + str(num_epochs) + 
        ' epochs, batch_size = ' + str(batch_size) +'.')
    for epoch in range(num_epochs) :

        for i in range(0, TRAIN_SIZE, batch_size) :
            
            slice = train_x[i:min(i+batch_size,TRAIN_SIZE)]

            #Get the set of predicted values from the model for each feature vector in the batch
            #Error happens at this line (v)
            predicted_y = model(slice)

            #Compute loss for this batch
            loss = loss_function(predicted_y, train_y[i:i+batch_size])
            
            #Do gradient descent 
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

        epoches.append(epoch)
        losses.append(loss.item())

        print('      * Epoch #' + str(epoch).rjust(8) + '\t| loss = ' + format(loss.item()).rjust(8) + '\t| time = ' + format(round(time.time()-prev_time,6)).ljust(6) + ' s.')
        if( loss.item() < loss_threshold ) :
            print(' -- Model training terminated by sufficiently low loss ( < ' + str(loss_threshold) + '  ):\n    max ' + 
                str(num_epochs) + ' epochs, batch_size = ' + str(batch_size) +'.')
            terminated_early = True
            break
        prev_time = time.time()
        
    if( not terminated_early ) : 
        print(' -- Model training terminated by epoch limit: ' + str(num_epochs) + 
        ' epochs, batch_size = ' + str(batch_size) +'.')


    #Move model to CPU now that we're done training, 
    #since even evaluating the entire dataset is pretty fast
    model.to(torch.device('cpu'))

    #Declare a variable to count how many test cases we correctly identify
    correct = 0

    #Test model accuracy
    for i in range(TEST_SIZE) :
        if( test_y[i].argmax() == model(torch.FloatTensor(test_x[i])).argmax() ) :
            correct += 1
    print(' -- MLP Model tested, correctly identitifies ' + str(100*correct/TEST_SIZE) + '% of ' 
        + str(TEST_SIZE) + ' test cases.\n' )

    #Save model w/ joblib 
    joblib.dump(model, 'model/trainedMLP')
    print(" -- MLP Model dumped with joblib\n")

    #Plot loss for visual
    plt.plot(epoches, losses)
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoches")

    plt.savefig("visualizations/MLP_visualization.png")

#Train the MLP model
MLP()

print("-------\nMLPTraining.py terminated successfully.\n----\n")