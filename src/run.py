import os

#Run DataLoading only if loaded is empty
if(os.listdir('loaded') == []) : import DataLoading

#Run DataPreprocessing only if preprocessed is empty
if(os.listdir('preprocessed') == []) : import DataPreprocessing

#Run ModelTraining
import ModelTraining

#Run Visualizing
import Visualizing