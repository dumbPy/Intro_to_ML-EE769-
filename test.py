#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dumbPy
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

def X_Cleaned(X, features):
    X =pd.get_dummies(X)
    X = X.loc[:, features]
    X = (X - X.mean())/(X.max()-X.min())    #X Normalized
    return X

def readCSV(filename, index_column):
    #Data imported in dataframe and Id column set to Index
    df = pd.read_csv(filename, index_col = index_column)
    df.isnull().any()
    #df = df.dropna()
    df = df.fillna(method='ffill')
    return df
 
def formDataFrame(bestOut):
    a = pd.DataFrame(X.index)
    a[list(y)[0]] = bestOut
    return a
    
def runTest(model, X):   
    model_name = str(model).split('.')[3].split("'")[0]
    clf, features = joblib.load('finalModel_'+model_name+'.pkl')
    #print(clf)
    #print(features)
    X = X_Cleaned(X, features)
    prediction = clf.predict(X)
    print('******************************************************')
    print('Model = '+model_name)
    acc_score = accuracy_score(y,  prediction)
    score.append(acc_score)
    print ("Test Accuracy     :: ", acc_score)
    print('******************************************************')
    print('\n\n')
    return(prediction)


#features_to_plot = ['OverallQual', 'GrLivArea']
models = [RandomForestClassifier, GradientBoostingClassifier, MLPClassifier]
score = []
X = readCSV('test.csv', index_column='Id')
y = readCSV('gt.csv', index_column='Id')

for model in models:
    runTest(model, X)
best_Classifier_index = score.index(max(score))
best_Classifier = models[best_Classifier_index]
best_Classifier_Name = str(best_Classifier).split('.')[3].split("'")[0]
print('Best Classifier by Accuracy Score : '+best_Classifier_Name)
bestOut = runTest(best_Classifier, X)
bestOut = formDataFrame(bestOut)
bestOut.to_csv('out.csv', index=False)
print('Prediction of '+best_Classifier_Name+' written to out.csv')
