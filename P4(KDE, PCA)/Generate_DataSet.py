#!/usr/bin/env python
# coding: utf-8

#DataSet_described
from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np

mu_s_1 = [[2, 5],
          [8, 1],
          [5, 3]]
cov_s_1 = [[[2, 0], [0, 2]],
           [[3, 1], [1, 3]],
           [[2, 1], [1, 2]]]

size= 500


#Generate_DataSet_function
def generate_dataset(mu_s, cov_s, label_sampels_size):
    dataset = pd.DataFrame(data={'X1': [], 'X2': [], 'Y': []})
    for i, mu_cov in enumerate(zip(mu_s, cov_s)):
        mu, cov = mu_cov
        x1, x2 = np.random.multivariate_normal(mu, cov, label_sampels_size).T
        temp = pd.DataFrame(
            data={'X1': x1, 'X2': x2, 'Y': [i]*label_sampels_size})
        dataset = pd.concat([dataset, temp], axis=0)
    return dataset


#Show_DataSet
dataset1 = generate_dataset(mu_s_1, cov_s_1, size)
dataset1.to_csv('dataset.csv', index=False)

dataset1

