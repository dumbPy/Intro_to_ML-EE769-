#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:03:52 2018

@author: tinkerman
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

def Xandy(Data):
    X = Data.iloc[:, 0:-1]
    X = pd.get_dummies(X)
    y = Data.iloc[:, -1]
    return X, y

def tune(testData, classifier, param_grid, verbose=0):
    """
    Input: testData
    Uses this testData to call randomsearchCV for tuning parameters.
    output: Trained model, best_parameters
    """
    
    #print('Tuning Parameters for : '+str(classifier))
    grid = param_grid
    
    X, y = Xandy(testData)
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

    X, y = Xandy(testData)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)

    # Train and Test dataset size details
    if verbose==True:
        print ("Train_x Shape :: ", X_train.shape)
        print ("Train_y Shape :: ", y_train.shape)
        print ("Test_x Shape :: ", X_test.shape)
        print ("Test_y Shape :: ", y_test.shape)

    if parameters:
        clf = classifier(**parameters)
    else:
        clf = classifier()
    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    y_header = list(y)
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

def crossValidation(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    print("Average CrossValidation Score of %0.2f runs: %0.5f\n" %(cv, scores.mean()))

def dtClassifier(testData, parameters=False, cvFlag = True, verbose=False, testSize = 0.2):
    """
    Support Vector Classifier
    Input:   testData, featureDropFlag(optional) to drop features
    Prints:  Accuracy score
    Returns: trained model clf, testData(features dropped)
    """
    print('******************************************************')

    X, y = Xandy(testData)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)

    # Train and Test dataset size details
    if verbose==True:
        print ("Train_x Shape :: ", X_train.shape)
        print ("Train_y Shape :: ", y_train.shape)
        print ("Test_x Shape :: ", X_test.shape)
        print ("Test_y Shape :: ", y_test.shape)

    if parameters:  clf = dtc(**parameters)
    else:           clf = dtc()
    
    clf = clf.fit(X_train, y_train)
    
    print ("Features Selected :: ", len(X.columns))
    print ("Train Accuracy    :: ", accuracy_score(y_train, clf.predict(X_train))*100)
    print ("Test Accuracy     :: ", accuracy_score(y_test,  clf.predict(X_test))*100)
    if cvFlag==True:    crossValidation(clf, X, y)
    
    testData = pd.concat([X, y], axis = 1)
    return(clf, testData)


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
    #print('Tuning Parameters.. ETA 1Mins... Wait....')
    
    param_grid_rf =    {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': [int(x) for x in np.linspace(2, 1000, num = 30)],
                        'min_samples_split': [int(x) for x in np.linspace(2, 20, num = 10)],
                        'min_samples_leaf': [int(x) for x in np.linspace(1, 10, num = 10)],
                        'bootstrap': [True, False]}
    
    bestParameters_rf = tune(testData, classifier=RandomForestClassifier, param_grid = param_grid_rf)
    print("Training Model after Feature Drop and Hyperparameter Tuning")
    final_model_rf = ensembleClassifier(testData, classifier=RandomForestClassifier, parameters=bestParameters_rf, cvFlag=True)
    final_features_rf = list(Xandy(testData)[0])
    testData = pd.get_dummies(df)
    
    
    print('******************************************************')
    print('Model = Gradient Boosting Classifier')
    
    print('******************************************************\n')
    #Fit, Test and Drop Features
    model, testData = ensembleClassifier(testData, classifier=GradientBoostingClassifier, featureDropFlag=True, featureSize=10, cvFlag=True,verbose=2)
    
    print("          Training after Feature Drop")
    model, testData = ensembleClassifier(testData, classifier=GradientBoostingClassifier, cvFlag=True)
    
    #Parameter Tuning for Gradient Boosting
    param_grid_gb =    {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 5)],
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': [int(x) for x in np.linspace(2, 1000, num = 5)],
                        'min_samples_split': [int(x) for x in np.linspace(2, 20, num = 5)],
                        'min_samples_leaf': [int(x) for x in np.linspace(1, 10, num = 5)],
                        }
    print('Tuning Parameters.. ETA 2Mins... Wait....')
    bestParameters_gb = tune(testData, classifier=GradientBoostingClassifier, param_grid = param_grid_gb, verbose=0)
    print("Training Model after Feature Drop and Hyperparameter Tuning")
    final_model_gb = ensembleClassifier(testData,classifier=GradientBoostingClassifier, parameters=bestParameters_gb, cvFlag=True)
    final_features_gb = list(Xandy(testData)[0])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print('******************************************************')
    print('                     Model = SVC'                      )
    print('******************************************************')
    """
    Hyperparameter tuning for SVM seemed to hang at kernel.
    With kernel set to default, accuracy is around 55% for unseen data (overfitting)
    Hence: default values of SVM were used-
    except for 'gamma' : np.exp(-16.75), that was found by manual binary search
    """
    final_model_svc = svClassifier(testData)
    
    print('******************************************************')
    print('Model =Decision Tree Classifier'                       )
    print('******************************************************')    
    model_dtc, testData = dtClassifier(testData, cvFlag=True, verbose=True)
    
    

# =============================================================================
#     param_grid_svc = {'C': [100,400,800],
#                       'kernel': ['linear','poly','rbf'],
#                       'degree': [3,4,5],
#                       'gamma': [0.1,0.01,0.001]}
#     
#     bestParameters_svc = tune(testData, classifier=SVC, param_grid = param_grid_svc, verbose=0)    
# =============================================================================




def svClassifier(testData, verbose=False, testSize = 0.2):
    X, y = Xandy(testData)
    params = {'C' : 10000, 'gamma' : np.exp(-16.75), 'degree': 3, 'kernel': 'rbf'}
    clf = SVC(**params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf.fit(X_train, y_train)
    print ("Train Accuracy    :: ", accuracy_score(y_train, clf.predict(X_train))*100)
    print ("Test Accuracy     :: ", accuracy_score(y_test,  clf.predict(X_test))*100)
    crossValidation(clf, X, y)
    return(clf)

main()










#best_params_rf = {'bootstrap': True, 'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
#best_params_rf = {'n_estimators': 1797, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 50, 'bootstrap': False}



#gives the list of all neighborhoods (returns array of unique elements from column 'Neighborhood')
#neighborhood = testData.Neighborhood.unique()