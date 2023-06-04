#!/usr/bin/python3

from sys import argv, exit

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge

inputs = np.load('inputs.npy')

inputs.shape

labels = np.load('labels.npy')
labels.shape

print(inputs)

X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)
X_train.shape

clf = LinearRegression().fit(X_train, y_train)

print("test mean accuracy:")
print(clf.score(X_test, y_test))

lso = Lasso(alpha=0).fit(X_train, y_train)

print("test mean accuracy:")
print(lso.score(X_test, y_test))

rdg = Ridge(alpha=1, fit_intercept=False).fit(X_train, y_train)


print("test mean accuracy:")
print(rdg.score(X_test, y_test))