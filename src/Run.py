import os

#Run DataLoading only if loaded is not properly filled
if(os.listdir('loaded') != ['train_x_loaded', 'num_testing_entries', 'test_x_loaded', 'train_y_loaded', 'test_y_loaded', 'num_training_entries']) : 
    import DataLoading

#Run DataPreprocessing only if preprocessed is not properly filled
if(os.listdir('preprocessed') != ['train_x_preprocessed', 'test_x_preprocessed']) : 
    import DataPreprocessing

#Run ModelTraining only if model is not properly filled
if( os.listdir('model') != ['trainedMLP', 'trainedSVM'] ) : 
    import ModelTraining
