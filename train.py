#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:03:52 2018

@author: dumbPy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import cross_val_score
from matplotlib.pyplot import subplots, show
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier


def Xandy(Data, verbose=False, testSize=0.2):
    X = Data.iloc[:, 0:-1]
    X = pd.get_dummies(X)
    X = (X - X.mean())/(X.max()-X.min())    #X Normalized
    y = Data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)

    # Train and Test dataset size details
    if verbose==True:
        print ("Train_x Shape :: ", X_train.shape)
        print ("Train_y Shape :: ", y_train.shape)
        print ("Test_x Shape :: ", X_test.shape)
        print ("Test_y Shape :: ", y_test.shape)
    return X, y, X_train, X_test, y_train, y_test 

def tune(testData, classifier, param_grid, verbose=0):
    """
    Input: testData
    Uses this testData to call randomsearchCV for tuning parameters.
    output: Trained model, best_parameters
    """
    
    #print('Tuning Parameters for : '+str(classifier))
    grid = param_grid
    
    X, y, X_train, X_test, y_train, y_test = Xandy(testData,verbose = verbose, testSize=0.2)
    clf = classifier()
    numericParameters = [i for i in grid.keys() if type(grid[i][0])==int]
    bestParameters = {}
    plot_rows = int(np.sqrt(len(numericParameters)))
    fig, axes = subplots(plot_rows, int(len(numericParameters)/plot_rows))
    ax = axes.flatten()
    for key in grid:
        grid2 = {key:grid[key]}
        Search = GridSearchCV(clf, param_grid=grid2, n_jobs=-1, return_train_score=True, cv=3, verbose=verbose)
        Search.fit(X, y)
        searchResults = pd.DataFrame(Search.cv_results_)
        bestParameters.update(Search.best_params_)
        if verbose:
            print(searchResults)

        if key in numericParameters:
            key_index = numericParameters.index(key)
            ax[key_index].plot(grid2[key], searchResults.loc[:, 'mean_test_score'])
            ax[key_index].set_xlabel("%s"%(key))
            ax[key_index].set_ylabel('Accuracy Score')
    show()
    print('******************************************************')
    print('Best Parameters decided by Parameter Tuning: ')
    print("")
    print(pd.DataFrame(list(bestParameters.items()), columns = ['Parameter', 'Value']))
    return (bestParameters)



def ensembleClassifier(testData, classifier, parameters=False, featureDropFlag = False, cvFlag = True, featureSize = 10, verbose=False, testSize = 0.2):
    """
    ensemble Classifier.
    Input:   testData, featureDropFlag(optional) to drop features
    Prints:  Accuracy score
    Returns: trained model clf, testData(features dropped)
    """
    print('******************************************************')

    X, y, X_train, X_test, y_train, y_test = Xandy(testData,verbose = verbose, testSize=testSize)

    if parameters:
        clf = classifier(**parameters)
    else:
        clf = classifier()
    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print ("Features Selected :: ", len(X.columns))
    print ("Train Accuracy    :: ", accuracy_score(y_train, clf.predict(X_train)))
    print ("Test Accuracy     :: ", accuracy_score(y_test, predictions))
    if cvFlag==True:
        crossValidation(clf, X, y)
    if featureDropFlag==True:
        print('******************************************************')
        print("      Dropping Features by feature_importances_")
    #Print Feature Importance
    head = ["name", "score"]
    featureImportances = sorted(zip(X_train.columns, clf.feature_importances_), key=lambda x: x[1] * -1)
    if verbose==True:
        print(pd.DataFrame(featureImportances, col = head))

    #update testData after feature Drop and return it.
    if featureDropFlag==True:
        X = X[list(pd.DataFrame(featureImportances[:featureSize])[0])]
    else:
        X = X[list(pd.DataFrame(featureImportances[:])[0])]
    testData = pd.concat([X, y], axis = 1)
    return(clf, testData)

def readCSV(filename, index_column):
    #Data imported in dataframe and Id column set to Index
    df = pd.read_csv("trainSold.csv", index_col = index_column)
    df.isnull().any()
    df = df.dropna()
    #df = df.fillna(method='ffill')
    return df

def dumpToFile(classifier, Data, fileName):
    feature_list = list(Xandy(Data)[0])
    joblib.dump((classifier, feature_list), fileName)

def crossValidation(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    print("Average CrossValidation Score of %0.2f runs: %0.5f\n" %(cv, scores.mean()*100))

def dtClassifier(testData, parameters=False, cvFlag = True, verbose=False, testSize = 0.2):
    """
    Support Vector Classifier
    Input:   testData, parameters(oprional)
    Prints:  Accuracy score
    Returns: trained model clf, testData(features dropped)
    """

    X, y, X_train, X_test, y_train, y_test = Xandy(testData,verbose = verbose, testSize=testSize)
    
    if parameters:  clf = dtc(**parameters)
    else:           clf = dtc()
    
    clf = clf.fit(X_train, y_train)
    
    print ("Features Selected :: ", len(X.columns))
    print ("Train Accuracy    :: ", accuracy_score(y_train, clf.predict(X_train))*100)
    print ("Test Accuracy     :: ", accuracy_score(y_test,  clf.predict(X_test))*100)
    if cvFlag==True:    crossValidation(clf, X, y)
    
    testData = pd.concat([X, y], axis = 1)
    return clf, testData

def mlpClassifier(testData, paramaters=False, verbose=False, testSize = 0.2):
    """
    using Neural Network Classifier
    Input:   testData, parameters(oprional)
    Prints:  Accuracy score
    Returns: trained model clf, testData(features dropped)
    """
    
    X, y, X_train, X_test, y_train, y_test = Xandy(testData,verbose = verbose, testSize=testSize)
    clf = MLPClassifier(max_iter=5000)
    clf.fit(X_train, y_train)
    print ("Features Selected :: ", len(X.columns))
    print ("Train Accuracy    :: ", accuracy_score(y_train, clf.predict(X_train))*100)
    print ("Test Accuracy     :: ", accuracy_score(y_test,  clf.predict(X_test))*100)
    crossValidation(clf, X, y)
    testData = pd.concat([X, y], axis = 1)
    return clf, testData
    

def svClassifier(testData, verbose=False, testSize = 0.2):
    params = {'C' : 10000, 'gamma' : np.exp(-16.75), 'degree': 3, 'kernel': 'rbf'}
    clf = SVC(**params)
    X, y, X_train, X_test, y_train, y_test = Xandy(testData,verbose = verbose, testSize=testSize)
    clf.fit(X_train, y_train)
    print ("Train Accuracy    :: ", accuracy_score(y_train, clf.predict(X_train))*100)
    print ("Test Accuracy     :: ", accuracy_score(y_test,  clf.predict(X_test))*100)
    crossValidation(clf, X, y)
    return clf
    

def main():
    df = readCSV("trainSold.csv", index_column = 'Id')

    #OneHot Encoding the data.
    testData = df   
    print('******************************************************')
    print('Model = Random Forest Classifier')
    print('******************************************************')  

    #Fit, Test and Drop Features
    model, testData = ensembleClassifier(testData,classifier=RandomForestClassifier, featureDropFlag=True, featureSize=10, verbose=False)
    
    print("          Training after Feature Drop")
    model, testData = ensembleClassifier(testData, classifier=RandomForestClassifier)
    
    #Parameter Tuning    
    param_grid_rf =    {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': [int(x) for x in np.linspace(2, 1000, num = 30)],
                        'min_samples_split': [int(x) for x in np.linspace(2, 20, num = 10)],
                        'min_samples_leaf': [int(x) for x in np.linspace(1, 10, num = 10)],
                        'bootstrap': [True, False]
                        }
    print('Tuning Parameters.. ETA 1Mins... Wait....')
    bestParameters_rf = tune(testData, classifier=RandomForestClassifier, param_grid = param_grid_rf)
    print("Training Model after Feature Drop and Hyperparameter Tuning")
    final_model_rf, testData = ensembleClassifier(testData, classifier=RandomForestClassifier, parameters=bestParameters_rf, cvFlag=True)
    dumpToFile(final_model_rf, testData, 'model_RForest.pkl')
    
    
    print('******************************************************')
    print('Model = Gradient Boosting Classifier')
    print('******************************************************\n')
    #Fit, Test and Drop Features
    testData = df
    model, testData = ensembleClassifier(testData, classifier=GradientBoostingClassifier, featureDropFlag=True, featureSize=10, cvFlag=True,verbose=2)
    
    print("          Training after Feature Drop")
    model, testData = ensembleClassifier(testData, classifier=GradientBoostingClassifier, cvFlag=True)
    
    #Parameter Tuning for Gradient Boosting
    param_grid_gb =    {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 20)],
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': [int(x) for x in np.linspace(1, 100, num = 5)],
                        'min_samples_split': [int(x) for x in np.linspace(2, 20, num = 10)],
                        'min_samples_leaf': [int(x) for x in np.linspace(1, 10, num = 5)],
                        }
    print('Tuning Parameters.. ETA 2Mins... Wait....')
    bestParameters_gb = tune(testData, classifier=GradientBoostingClassifier, param_grid = param_grid_gb, verbose=0)
    print("Training Model after Feature Drop and Hyperparameter Tuning")
    #final_model_gb, testData = ensembleClassifier(testData,classifier=GradientBoostingClassifier, parameters=bestParameters_gb, cvFlag=True, testSize=0.33)
    final_model_gb, testData = ensembleClassifier(testData, classifier=GradientBoostingClassifier, cvFlag=True)
    dumpToFile(final_model_gb, testData, 'model_GBoosting.pkl')
    
    """
    To Use TestData cleaned by above Model
    """
    df=testData
    print('\n  Training on 10 best Features from Here On.\n')
    print('******************************************************')
    print('                     Model = SVC'                      )
    print('******************************************************')
    """
    Hyperparameter tuning for SVM seemed to hang at kernel.
    With kernel set to default, accuracy is around 55% for unseen data (overfitting)
    Hence: default values of SVM were used-
    except for 'gamma' : np.exp(-16.75), that was found by manual binary search
    """
    testData=df
    final_model_svc = svClassifier(testData)
    dumpToFile(final_model_svc, testData, 'model_SVC.pkl')
    print('******************************************************')
    print('Model =Decision Tree Classifier'                       )
    print('******************************************************')    
    testData=df
    final_model_dtc, testData = dtClassifier(testData, cvFlag=True, verbose=False)
    dumpToFile(final_model_dtc, testData, 'model_DTC.pkl')
    
    print('******************************************************')
    print('       Model = Multi Layer Perceptron')
    print('******************************************************')    
    testData=df
    final_model_mlp, testData = mlpClassifier(testData)
    dumpToFile(final_model_mlp, testData, 'model_MLP.pkl')
    

main()


#gives the list of all neighborhoods (returns array of unique elements from column 'Neighborhood')
#neighborhood = testData.Neighborhood.unique()