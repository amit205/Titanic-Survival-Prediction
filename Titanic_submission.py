# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 17:17:45 2018

@author: amit
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset. Importing Pclass, Sex, Age, SibSp, Parch columns for train.csv and test.csv. 
# Also importing Survived column in variable y.
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5,6,7]].values
y = dataset.iloc[:, 1].values

testdataset = pd.read_csv('test.csv')
X_test = testdataset.iloc[:, [1,3,4,5,6]].values
passengerId = testdataset.iloc[:,0]

# Replacing the missing values in Age column with mean of the age column
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_train = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer_train.transform(X[:,2:3])

imputer_test = imputer.fit(X_test[:, 2:3])
X_test[:, 2:3] = imputer_test.transform(X_test[:,2:3])

# Taking care of categorical data (sex)
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:,1])

X_test[:, 1] = labelencoder_X.fit_transform(X_test[:,1])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Fitting Random Forest Classifier to the training set
"""from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0)
classifier.fit(X, y)"""

# Fitting Logistics Regression classifier to the training set
"""from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y)"""

# Fitting KNeighbors Classifier to the training set
"""from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 100, metric = 'minkowski', p = 2, weights = 'distance')
classifier.fit(X, y)"""

# Fitting gaussian kernel SVM to the training set. It gives the best results. Hence, uncommented.
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, y)

# Fitting Naive Bayes classifier to the training set
"""from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)"""

# Fitting Linear SVM to the Training set
"""from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X, y)"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Survived']
results = pd.concat([passengerId, y_pred], axis = 1)
results.to_csv('results.csv', index = False)
