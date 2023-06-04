#!/usr/bin/python3

from sys import argv, exit

import numpy as np

array = np.zeros((300,6))
array.shape

array[:,0] = np.random.normal(2.5,1, (300))
np.mean(array, axis=0)

array[:,1] = np.random.randint(20,60,(300))
print(array)

array[:, 2] = np.random.normal(200, 50, 300)
array[:, 2].sort()

array[:, 3] = np.random.normal(300, 90, 300)
array[:, 3][::-1].sort()
print(np.corrcoef(array[:, 2], array[:, 3]))

array[:, 4] = np.random.normal(400, 200, 300)
array[:, 4].sort()
print(np.corrcoef(array[:, 2], array[:, 4]))

array[:, 5] = np.random.normal(700, 900, 300)
print(np.corrcoef(array[:, 2], array[:, 5]))

np.mean(array, axis=0)

array.std(axis=0)

np.savetxt("artificial_dataset.csv", array, delimiter=",")

exit(0)