#!/usr/bin/python3

from sys import argv, exit

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.svm import LinearSVC

dataframe = pd.read_csv("dataset.csv")
dataframe.info()

dataframe.head(10)

dataframe['quality'].plot(kind='hist').set(xlabel="quality")

dataframe['quality'] = dataframe['quality'].apply(lambda x: 0 if x < 7 else 1 )
sns.countplot(x='quality', data=dataframe)

plt.matshow(dataframe.corr())
plt.show()

dataframe.corr()

dataframe_scaled = pd.DataFrame(MinMaxScaler().fit_transform(dataframe))
sns.boxplot(dataframe_scaled)

sns.boxplot(dataframe['fixed acidity']).set(ylabel = "quantity",xlabel = "fixed acidity")

sns.boxplot(dataframe['volatile acidity']).set(ylabel = "quantity",xlabel = "Volatile acidity")

sns.boxplot(dataframe['citric acid']).set(ylabel = "quantity",xlabel = "Citric acid")

sns.boxplot(dataframe['residual sugar']).set(ylabel = "quantity",xlabel = "Residual sugar")

sns.boxplot(dataframe['chlorides']).set(ylabel = "quantity",xlabel = "Chlorides")

sns.boxplot(dataframe['free sulfur dioxide']).set(ylabel = "quantity",xlabel = "free sullfure dioxide")

sns.boxplot(dataframe['total sulfur dioxide']).set(ylabel = "quantity",xlabel = "Total sulfur dioxide")

sns.boxplot(dataframe['density']).set(ylabel = "density")

sns.boxplot(dataframe['pH']).set(ylabel = "pH")

sns.boxplot(dataframe['sulphates']).set(ylabel = "quantity",xlabel = "Sulphates")

sns.boxplot(dataframe['alcohol']).set(ylabel = "quantity",xlabel = "Alcohol")

dataframe = dataframe.drop(dataframe[(dataframe['fixed acidity']>15)].index)
dataframe = dataframe.drop(dataframe[(dataframe['volatile acidity']>1.4)].index)
dataframe = dataframe.drop(dataframe[(dataframe['citric acid']>0.9)].index)
dataframe = dataframe.drop(dataframe[(dataframe['residual sugar']>15)].index)
dataframe = dataframe.drop(dataframe[(dataframe['chlorides']>0.5)].index)
dataframe = dataframe.drop(dataframe[(dataframe['free sulfur dioxide']>70)].index)
dataframe = dataframe.drop(dataframe[(dataframe['total sulfur dioxide']>250)].index)

dataframe = dataframe.drop(dataframe[(dataframe['pH']>4)].index)
dataframe = dataframe.drop(dataframe[(dataframe['pH']<2.8)].index)
dataframe = dataframe.drop(dataframe[(dataframe['sulphates']>1.75)].index)
dataframe = dataframe.drop(dataframe[(dataframe['alcohol']>14.5)].index)

dataframe.shape

plt.matshow(dataframe.corr())
plt.show()

X = dataframe.drop(columns=['quality'])
y = dataframe['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train.ravel())

score = neigh.score(X_test, y_test)
print("Score: ", score)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

print("test mean accuracy:")
print(clf.score(X_test, y_test))

lsvc = LinearSVC(verbose=0, dual=False, C=1, multi_class='ovr', fit_intercept=False)
lsvc.fit(X_train, y_train)

score = lsvc.score(X_test, y_test)
print("Score: ", score)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

score = rfc.score(X_test, y_test)
print("Score: ", score)

