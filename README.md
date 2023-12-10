# SAT-4 Image Classification Project

Welcome to this repository! This project focuses on high-accuracy image classification using Multilayer-Perceptron (MLP) and Support Vector Machine (SVM) models on the [SAT-4 Airborne Dataset](https://www.kaggle.com/crawford/deepsat-sat4) from Kaggle.

## Setup Instructions

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/PeterFavero/SAT-4_Image_Classification_ML.git
   ```
   
2. Navigate to the project directory:

   ```bash
   cd SAT-4_Image_Classification_ML
   ```

3. Activate the virtual environment:

   - On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   
5. Download the [SAT-4 Airborne Dataset](https://www.kaggle.com/crawford/deepsat-sat4) from Kaggle and place the resultant 'archive' folder in the project's working directory.
   
Setting up your clone of the repository will come preloaded with two highly accurate classification models, model/trainedMLP, which achieves 99.711% accuracy on the test dataset, and model/trainedSVM, which achieves 99.845% accuracy on the test dataset. Further, a small training dataset of 10000 entries and a small testing dataset of 2500 entries will be stored unprocessed in directory 'loaded' and with x values stored as preprocessed feature vectors in directory 'preprocessed.' 

## Usage Instructions

### DataLoading.py 
Running DataLoading.py will load the entire training and testing dataset to be preprocessed and dump it using joblib into directory 'loaded.' If you want to load only part of the dataset for a specific run, change the values of num_training_entries and num_testing_entries on lines 46 and 47 accordingly. 

### DataPreprocessing.py
Running DataPreprocessing.py will preprocess loaded/train_x_loaded and loaded/test_x_loaded, the x values of the loaded training and testing datasets in 'loaded,' and then dump the resultant feature vectors into directory 'preprocessed.'

### ModelTraining.py
Running ModelTraining.py will train new MLP and SVM models on the current contents of directories 'loaded' and 'preprocessed' and produce visualizations for each model as well. This script will take a long time to complete if the full training and testing datasets are loaded, as it trains an MLP model to cross-entropy loss < 0.001, and then trains 6 different SVM models using 6 different C values. Therefore, if you want to expirement with tweaks to hyperparameters and/or have the model training go quickly either: 
   - do not run DataLoading.py before running ModelTraining after you clone the repo,
   - run the commands ```git restore loaded``` and then ```git restore preprocessed``` to restore the small training and testing datasets from the repo if you've loaded and/or preprocessed excessively large datasets into 'loaded' and 'preprocessed,'
   - or go into DataLoading.py and reduce the size of num_training_entries and num_testing_entries significantly (10000 and 2500 respectively are good places to start), and then run RunAll.py to update 'loaded' and 'preprocessed.'

### ModelTesting.py
Running ModelTesting.py will print the accuracies of model/trainedMLP and model/trainedSVM at classifying the loaded testing dataset as a percentage without retraining either model, which you can use to verify the accuracy of our preloaded models or your own. 

### RunAll.py
Running RunAll.py will run DataLoading.py, then DataPreprocessing.py, then ModelTraining.py, then ModelTesting.py.
