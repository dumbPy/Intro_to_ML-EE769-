import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from tabulate import tabulate   

def selectKImportance(model, X, k=5):
     return X[:,model.feature_importances_.argsort()[::-1][:k]]












#Data imported in dataframe and Id column set to Index
df = pd.read_csv("trainSold.csv", index_col = 'Id')
df.isnull().any()
df = df.dropna()
df = df.fillna(method='ffill')

# =============================================================================
# #dropped as the 'GarageArea' covers the size of the Garage.
# columns_to_drop = ['GarageCars']
# df.drop(columns_to_drop, axis = 1, inplace = True)
# 
# =============================================================================
#OneHot Encoding the data.
testData = pd.get_dummies(df)

#collects all the headers into seperate list before converting the data to numpy array

headers = list(testData)

X = testData.iloc[:, 0:-3]
y = testData.iloc[:, -3:]
#X = X.values
#y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and Test dataset size details
print ("Train_x Shape :: ", X_train.shape)
print ("Train_y Shape :: ", y_train.shape)
print ("Test_x Shape :: ", X_test.shape)
print ("Test_y Shape :: ", y_test.shape)


clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


print ("Train Accuracy :: ", accuracy_score(y_train, clf.predict(X_train)))
print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))

#Print Feature Importance
head = ["name", "score"]
featureImportances = sorted(zip(X_train.columns, clf.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(featureImportances, head, tablefmt="plain"))

X = X[list(pd.DataFrame(featureImportances[:10])[0])]

testData = pd.concat([X, y], axis = 1)



















#gives the list of all neighborhoods (returns array of unique elements from column 'Neighborhood')
#neighborhood = testData.Neighborhood.unique()