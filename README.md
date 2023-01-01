# Sensorless-Drive-Diagnosis
This repository contains a machine learning pipeline for classifying the quality of sensorless drives based on various features. The pipeline is implemented in Python using the pandas, numpy, sklearn, and matplotlib libraries.

# Data
The data for this project is stored in the file Datasets/Sensorless_drive_diagnosis.txt. It contains X number of features for Y number of instances, with a class label for each instance indicating the quality of the sensorless drive.

# Pipeline
The pipeline performs the following steps:
1. Read in the data from the file and do some basic data exploration (e.g. generate summary statistics, check for class balance, get information about the dataframe).
2. Split the data into training and testing sets, using 80% of the data for training and 20% for testing.
3. Train a decision tree classifier and use it to select important features from the training data.
4. Standardize the data using the training data.
5. Train and test a decision tree classifier, a multi-layer perceptron classifier, and a random forest classifier using the training and testing sets.
6. Plot the performance of the models using a confusion matrix and generate a classification report for each model.
7. Save the trained models to files using pickle.

# Requirements
To run this pipeline, you will need the following libraries:
    pandas
    numpy
    sklearn
    matplotlib
In addition, the VisualizeNN_mod is provided in this repository and must be placed in the same directory as the other code files.

# Directory Structure
The directory structure for this repository is as follows:

Sensorless Drive Diagnosis
├── classification_main_v1.py
├── Datasets
│   ├── Sensorless_drive_diagnosis.txt
└── VisualizeNN_mod.py

# Usage
To run the pipeline, simply run the script classification_pipeline.py using Python. The trained models will be saved to the Models directory.

# Results
The performance of the models will be printed to the console, including the confusion matrix and classification report for each model. The models will also be saved to the Models directory.

# Additional Notes
The classification models were implemented using scikit-learn version 0.23.1.
The VisualizeNN_mod module was modified from the original version by Milo Harper (https://github.com/miloharper/visualise-neural-network) for use in this project.
