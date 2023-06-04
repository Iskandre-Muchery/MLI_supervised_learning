#!/usr/bin/python3

from sys import argv, exit

import pandas as pd
import numpy as np
import math

dataframe = pd.read_csv("dataset.csv")
dataframe.info()

print(dataframe["favorite music style"].unique())
print(dataframe["city"].unique())
print(dataframe["job"].unique())

cites = pd.DataFrame({
    'paris':    [0.0    , 0.661, 0.588, 2.0   , 0.217],
    'marseille':[0.661, 0.0    , 0.319, 2.0   , 1    ],
    'toulouse': [0.588, 0.319, 0.0    , 2.0   , 0.894],
    'madrid':   [2.0   , 2.0   , 2.0   , 0   , 2.0   ],
    'lille':    [0.217, 1    , 0.894, 2.0   , 0.0    ]
    })
cites

def get_city_idx(city):
    i = 0
    for city in cites:
        if city.lower() == city:
            break
        i += 1
    return i

def get_city_diff(city_1, city_2):
    return cites.loc[get_city_idx(city_1)][get_city_idx(city_2)]

def get_music_diff(music_1, music_2):
    if music_1 == music_2:
        return 0
    if "metal" in music_1 and "metal" in music_2:
        return 0.5
    return 1

def get_job_diff(job_1, job_2):
    if job_1 == job_2:
        return 0
    return 1

def get_diff_number(num_1, num_2):
    return (math.sqrt((num_1 - num_2)**2))

def get_diff(p_1, p_2):
    age_diff = get_diff_number(dataframe.loc[p_1][1], dataframe.loc[p_2][1]) 
    height_diff = get_diff_number(dataframe.loc[p_1][2], dataframe.loc[p_2][2]) 
    job_diff = get_job_diff(dataframe.loc[p_1][3], dataframe.loc[p_2][3]) 
    city_diff = get_city_diff(dataframe.loc[p_1][4], dataframe.loc[p_2][4]) 
    music_diff = get_music_diff(dataframe.loc[p_1][5], dataframe.loc[p_2][5])
    return age_diff + height_diff + job_diff + city_diff + music_diff

nb_person = len(dataframe.index)
dissimilarity_matrix = np.zeros((nb_person, nb_person))
print("compute dissimilarities")
for p_1 in range(nb_person):
    for p_2 in range(nb_person):
        dissimilarity = get_diff(p_1, p_2)
        dissimilarity_matrix[p_1, p_2] = dissimilarity

print("dissimilarity matrix")
print(dissimilarity_matrix)

np.save('metric.npy', dissimilarity_matrix)

exit(0)
