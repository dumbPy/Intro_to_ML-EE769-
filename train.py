import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

seed = np.random.randint(500)

def Xandy(Data):
    X = Data.iloc[:, 0:-3]
    y = Data.iloc[:, -3:]
    return X, y

def tuneRF(testData):
    """
    Input: testData
    Uses this testData to call randomsearchCV for tuning parameters.
    output: Trained model, best_parameters
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 30, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    X, y = Xandy(testData)
    clf = RandomForestClassifier()
    Search = GridSearchCV(clf, param_grid=grid,n_jobs=-1, return_train_score=True, cv=3, verbose=2)
    Search.fit(X, y)
    SearchResults = pd.DataFrame(Search.cv_results_)
    bestParameters = Search.best_params_
    print(SearchResults)
    print(bestParameters)
    print(Search.best_score_)
    plt.plot(SearchResults.loc[:, 'param_n_estimators'], SearchResults.loc[:, 'mean_test_score'])
    plt.show()



def rfClassifier(testData, parameters=False, featureDropFlag = False, cvFlag = False, tuningFlag=False, featureSize = 10, printStatsFlag=False, testSize = 0.2):
    """
    Random Forest Classifier.
    Input:   testData, featureDropFlag(optional) to drop features
    Prints:  Accuracy score
    Returns: trained model clf, testData(features dropped)
    """
    #collects all the headers into seperate list(Useful only if data is converted to numpy, losing the column Names)
    headers = list(testData)
    X, y = Xandy(testData)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state = seed)

    # Train and Test dataset size details
    if printStatsFlag==True:
        print ("Train_x Shape :: ", X_train.shape)
        print ("Train_y Shape :: ", y_train.shape)
        print ("Test_x Shape :: ", X_test.shape)
        print ("Test_y Shape :: ", y_test.shape)

    if parameters:
        clf = RandomForestClassifier(parameters)
    else:
        clf = RandomForestClassifier()
    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print ("Features Selected :: ", len(X.columns))
    print ("Train Accuracy    :: ", accuracy_score(y_train, clf.predict(X_train)))
    print ("Test Accuracy     :: ", accuracy_score(y_test, predictions))
    if cvFlag==True:
        crossValidation(clf, X, y)
    if featureDropFlag==True:
        print("**************Dropping Features************")
    #Print Feature Importance
    head = ["name", "score"]
    featureImportances = sorted(zip(X_train.columns, clf.feature_importances_), key=lambda x: x[1] * -1)
    if printStatsFlag==True:
        print(tabulate(featureImportances, head, tablefmt="plain"))

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
    print("Average CrossValidation Score of %0.2f runs: %0.5f" %(cv, scores.mean()))



def main():
    df = readCSV("trainSold.csv", index_column = 'Id')

    #OneHot Encoding the data.
    testData = pd.get_dummies(df)
    #Fit, Test and Drop Features
    model, testData = rfClassifier(testData, featureDropFlag=True, cvFlag=True, featureSize=10, printStatsFlag=False)
    
    #Refitting the model after feature drop
    model, testData = rfClassifier(testData, cvFlag=True)
    
    #Parameter Tuning
    model, testData = rfClassifier(testData, cvFlag=True)
    
    tuneRF(testData)




main()












#best_params_rf = {'n_estimators': 1797, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 50, 'bootstrap': False}



#gives the list of all neighborhoods (returns array of unique elements from column 'Neighborhood')
#neighborhood = testData.Neighborhood.unique()