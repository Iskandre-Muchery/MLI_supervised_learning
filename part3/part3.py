#!/usr/bin/python3

from sys import argv, exit

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

inputs = np.load('inputs.npy')
labels = np.load('labels.npy')

X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

# Logistic Regression
logistic_params = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
logistic_clf = LogisticRegression(random_state=0)
logistic_grid_search = GridSearchCV(logistic_clf, logistic_params)
logistic_grid_search.fit(X_train, y_train.ravel())

logistic_best_params = logistic_grid_search.best_params_
logistic_best_score = logistic_grid_search.best_score_

print("Logistic Regression:")
print("Best Parameters:", logistic_best_params)
print("Best Score:", logistic_best_score)

# Linear SVM
svm_params = {'C': [0.1, 1, 10]}
svm_clf = LinearSVC(verbose=0)
svm_grid_search = GridSearchCV(svm_clf, svm_params)
svm_grid_search.fit(X_train, y_train.ravel())

svm_best_params = svm_grid_search.best_params_
svm_best_score = svm_grid_search.best_score_

print("Linear SVM:")
print("Best Parameters:", svm_best_params)
print("Best Score:", svm_best_score)

# k-Nearest Neighbors
knn_params = {'n_neighbors': [3, 5, 7]}
knn_clf = KNeighborsClassifier()
knn_grid_search = GridSearchCV(knn_clf, knn_params)
knn_grid_search.fit(X_train, y_train.ravel())

knn_best_params = knn_grid_search.best_params_
knn_best_score = knn_grid_search.best_score_

print("k-Nearest Neighbors:")
print("Best Parameters:", knn_best_params)
print("Best Score:", knn_best_score)

exit(0)