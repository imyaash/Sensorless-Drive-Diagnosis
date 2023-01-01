# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:02:51 2022

@author: imyaash-admin
"""

"""Importing necessary modules"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import accuracy_score as accs, confusion_matrix as con_mat, classification_report as report, ConfusionMatrixDisplay as CMD
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_selection import SelectFromModel as sfm
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neural_network import MLPClassifier as mlpc
from sklearn.ensemble import RandomForestClassifier as rfc
import VisualizeNN_mod as vnn
from sklearn.tree import plot_tree as pt
from random import choice
import pickle as pkl

"""Importing the dataset"""
df = pd.read_csv("Datasets/Sensorless_drive_diagnosis.txt", sep = " ", header = None)

#Renaming columns as V1, V2,...., Class
def colNames(df):
    col = []
    for i in range(1, len(list(df.columns))):
        col.append("V" + str(i))
    col.append("Class")
    col = dict(map(lambda x, y: (x, y), list(df.columns), col))
    df = df.rename(columns = col)
    return df

df = colNames(df)

#Basic exploration
df.describe() #Generate summary statistics for numerical columns
df["Class"].value_counts() #Checking for class balance
df.info() #Get information about the dataframe, such as the data types of each column
df.corr() #Generate a correlation matrix for the dataframe

for i in df.columns: #Loop through each column in the dataframe
    plt.plot(df[i]) #Plot the values in the column
    plt.show() #Show the plot

x =  df.drop("Class", axis = 1) #Drop the "Class" column from the dataframe and assign the rest to the "x" variable
y = df["Class"] #Assign the "Class" column to the "y" variable

trainX, testX, trainY, testY = tts(x, y, test_size = 0.2, random_state = 99) #Split the data into training and testing sets, using 80% of the data for training and 20% for testing

clf = dtc(criterion = "entropy", random_state = 99) #Create a DecisionTreeClassifier object
fetSel = sfm(estimator = clf) #Create a SelectFromModel object using the DecisionTreeClassifier as the estimator
fetSel.fit(trainX, trainY) #Train the model and select the important features using the training data
fetSel.estimator_.feature_importances_ #Access the feature importances of the DecisionTreeClassifier
plt.plot(x.columns.tolist(), fetSel.estimator_.feature_importances_) #Plot the feature importances along with the names of the columns (features)
fetSel.threshold_ #Print the threshold used by SelectFromModel to determine which features to keep
pd.concat([pd.Series(x.columns.tolist(), name = "Features"), pd.Series(fetSel.get_support(), name = "Support")], axis = 1)[pd.concat([pd.Series(x.columns.tolist(), name = "Features"), pd.Series(fetSel.get_support(), name = "Support")], axis = 1)["Support"] == True] #Concatenate the columns and the support list into a dataframe, then select only the rows where support is True
x = fetSel.transform(x) #Use the SelectFromModel object to select only the supported features from the original "x" data

trainX, testX, trainY, testY = tts(x, y, test_size = 0.2, random_state = 99) #Split the data into training and testing sets again
scaler = ss(copy = False) #Create a new StandardScaler object
trainX_scaled = scaler.fit_transform(trainX) #Fit the scaler to the training data and transform it
testX_scaled = scaler.transform(testX) #Transform the testing data using the fitted scaler

"""
hL = [6, 8, 12, 16, 24] #List of different numbers of nodes to use in the hidden layer
alpha = [] #List to store the number of nodes used in each model
accuracy = [] #List to store the accuracy of each model
for i in hL: #Loop through each number of nodes
    clf = mlpc(activation = "relu", solver = "lbfgs",
               hidden_layer_sizes = i,   #Set the number of nodes to the current value of "i"
               max_iter = 100000, verbose = True,
               warm_start = True, random_state = 99).fit(trainX_scaled, trainY) #Train the model on the scaled training data
    alpha.append(i) #Store the number of nodes in the "alpha" list
    accuracy.append(accs(testY, clf.predict(testX))) #Evaluate the model's accuracy on the test data and store it in the "accuracy" list
log = pd.concat([pd.Series(alpha, name = "Nodes"), pd.Series(accuracy, name = "Accuracy")], axis = 1) #Concatenate the "alpha" and "accuracy" lists into a dataframe
print(log[log["Accuracy"] == max(log["Accuracy"])]["Nodes"]) #Select the row with the highest accuracy and print the number of nodes

alphas = np.linspace(0.000001, 1, 5) #Generate 5 values for "alpha" in the range 0.000001 to 1
alpha = [] #List to store the "alpha" value used in each model
accuracy = [] #List to store the accuracy of each model
for i in alphas: #Loop through each "alpha" value
    clf = mlpc(activation = "relu", solver = "lbfgs",
               alpha = i, #Set the "alpha" value to the current value of "i"
               hidden_layer_sizes = 24,   #Set the number of nodes to 24
               max_iter = 100000, verbose = True,
               warm_start = True, random_state = 99).fit(trainX_scaled, trainY) #Train the model on the scaled training data
    alpha.append(i) #Store the "alpha" value in the "alpha" list
    accuracy.append(accs(testY, clf.predict(testX))) #Evaluate the model's accuracy on the test data and store it in the "accuracy" list
log = pd.concat([pd.Series(alpha, name = "Alpha"), pd.Series(accuracy, name = "Accuracy")], axis = 1) #Concatenate the "alpha" and "accuracy" lists into a dataframe
print(log[log["Accuracy"] == max(log["Accuracy"])]["Alpha"]) #Select the row with the highest accuracy and print the "alpha" value

alphas = np.linspace(0.0000001, 0.00001, 5) #Generate 5 values for "alpha" in the range 0.0000001 to 0.00001
alpha = [] #List to store the "alpha" value used in each model
accuracy = [] #List to store the accuracy of each model
for i in alphas: #Loop through each "alpha" value
    clf = mlpc(activation = "relu", solver = "lbfgs",
               alpha = i, #Set the "alpha" value to the current value of "i"
               hidden_layer_sizes = 24,   #Set the number of nodes to 24
               max_iter = 100000, verbose = True,
               warm_start = True, random_state = 99).fit(trainX_scaled, trainY) #Train the model on the scaled training data
    alpha.append(i) #Store the "alpha" value in the "alpha" list
    accuracy.append(accs(testY, clf.predict(testX))) #Evaluate the model's accuracy on the test data and store it in the "accuracy" list
log = pd.concat([pd.Series(alpha, name = "Alpha"), pd.Series(accuracy, name = "Accuracy")], axis = 1) #Concatenate the "alpha" and "accuracy" lists into a dataframe
print(log[log["Accuracy"] == max(log["Accuracy"])]["Alpha"]) #Select the row with the highest accuracy and print the "alpha" value
"""

MLPclf = mlpc(activation = "relu", solver = "lbfgs", alpha = 0.00001,
              hidden_layer_sizes = 24, max_iter = 10000, verbose = True,
              warm_start = True, random_state = 99) #Create an instance of the MLPClassifier with the specified hyperparameters
MLPclf.fit(trainX_scaled, trainY) #Train the model on the scaled training data
pkl.dump(MLPclf, open("Models/MLPclf.pkl", "wb")) #Save the trained model to a file using pickle
MLPpredY = MLPclf.predict(testX_scaled) #Use the trained model to make predictions on the scaled test data

"""
estimator = [] #List to store the number of estimators used in each model
accuracy = [] #List to store the accuracy of each model
for i in np.arange(100, 1100, 100): #Loop through different values for the number of estimators (100, 200, 300, ..., 1000)
    clf = rfc(n_estimators = i, criterion = "entropy", n_jobs = -1, verbose = 1, random_state = 99) #Create an instance of the RandomForestClassifier with the specified number of estimators and other hyperparameters
    clf.fit(trainX, trainY) #Train the model on the training data
    estimator.append(i) #Store the number of estimators in the "estimator" list
    accuracy.append(accs(testY, clf.predict(testX))) #Evaluate the model's accuracy on the test data and store it in the "accuracy" list
log = pd.concat([pd.Series(estimator, name = "Estimator"), pd.Series(accuracy, name = "Accuracy")], axis = 1) #Concatenate the "estimator" and "accuracy" lists into a dataframe
"""

RFclf = rfc(n_estimators = 600, # number of trees in the forest
            criterion = "entropy", # splitting criterion
            n_jobs = -1, # number of parallel jobs to run
            verbose = 3, # level of verbosity of the output
            random_state = 99) # random seed

# Fit the classifier on the training data
RFclf.fit(trainX, trainY)

# Pickle and save the trained classifier
pkl.dump(RFclf, open("Models/RFclf.pkl", "wb"))

# Use the trained classifier to make predictions on the test data
RFpredY = RFclf.predict(testX)


"""Cross Evaluating the models"""
MLPCacc = accs(testY, MLPpredY) #0.0.9796615963083234 #Calculating the accuracy for the Neural Network model
print("Neural Network Accuracy:", MLPCacc)
RFCacc = accs(testY, RFpredY) #0.9830798154161682 #Calculating the accuracy for the Random Forest model
print("Random Forest Accuracy", RFCacc)
MLPCresult = report(testY, MLPpredY) #Getting classification model report for the the Neural Network model
print("Neural Network Classification Report:")
print(MLPCresult)
RFCresult = report(testY, RFpredY) #Getting classification model report for the Random Forest model
print("Random Forest Classification Report:")
print(RFCresult)
MLPCcm = con_mat(testY, MLPpredY) #Calculating a confusion matrix for the the Neural Network model
RFCcm = con_mat(testY, RFpredY) #Calculating a confusion matrix for the Random Forest model
MLPCcmd = CMD(confusion_matrix = MLPCcm, display_labels = MLPclf.classes_) #Creating a Confusion Matrix Model Plot for the the Neural Network model
MLPCcmd.plot(cmap = "Greys", include_values = False) #Showing the Confusion Matrix Plot for the the Neural Network model
RFCcmd = CMD(confusion_matrix = RFCcm, display_labels = RFclf.classes_) #Creating a Confusion Matrix Model Plot for the Random Forest model
RFCcmd.plot(cmap = "Greys", include_values = False) #Showing the Confusion Matrix Plot for the Random Forest model

#Visualising difference between (Random Forest Classifier and MultiLayer Perceptor Neural Network) Models
diffMat = RFCcm - MLPCcm
ax = sns.heatmap(diffMat, cmap = plt.cm.Blues, annot = True)
plt.xlabel("Predicted Label", fontdict = {"family": "Perpetua", "color": "#272f4d", "size": 16})
plt.ylabel("True Label", fontdict = {"family": "Perpetua", "color": "#272f4d", "size": 16})
plt.title("Diffrence b/w RF & MLPNN Model", fontdict = {"family": "Perpetua", "color": "#272f4d", "size": 18})
plt.show()

"""Visualising the Models"""
netStr = np.hstack(([trainX.shape[1],
                     np.asarray(MLPclf.hidden_layer_sizes),
                     len(np.unique(trainY))])) #Creating Neural Network Structure
net = vnn.DrawNN(netStr) #Drawing Neural Network Structure
net.draw() #Plotting the Neural Network


fig = plt.figure(figsize = (16, 9), dpi = 800) #Setting figure size
fig.patch.set_facecolor("slategray") #Setting figure backgraound colour
pt(choice(RFclf.estimators_), feature_names = df.drop("Class", axis = 1).columns, class_names = str(np.unique(trainY)),
   rounded = True, proportion = False, precision = 2, filled = True) #plotting a random tree from the forrest
fig.savefig("TreeFromForest.png") #Saving the plot
