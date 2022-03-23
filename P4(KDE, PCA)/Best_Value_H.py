#!/usr/bin/env python
# coding: utf-8

#Show_DataSet
import numpy as np
import pandas as pd
import math
from math import sqrt
from operator import add
from functools import reduce
from scipy.stats import multivariate_normal


def raw_data(path):
    data = pd.read_csv(path)
    return data.values


#Gaussian_function
def gaussian(u, sigma):
      return (1/(math.sqrt(2*math.pi) * sigma)) * math.exp(-(u**2)/(2*sigma**2))


#L2_Norm_function
def l2(d1, d2):
    return sqrt(reduce(add, map(lambda y1, y2: (y1 - y2) ** 2, d1, d2)))


#TruePDF_function
def truePDF(data, mu, cov, per):
    true_pdf = 0
    for i in range(len(mu)):
        true_pdf += (per[i] * multivariate_normal(mu[i], cov[i]).pdf(data))
    return true_pdf


#Kernel_function
def kernel(dataPoint, x, h, sigma):
    dim = len(dataPoint)
    prod = 1
    for j in range(0, dim):
        prod *= gaussian((x[j]-dataPoint[j])/h, sigma)
    return prod


#KFold_function
def kFold(data_length, k):
    folds = []
    fold_size = data_length/k
    for i in range(k):
        folds.append([int(i*fold_size),int(((i+1)*fold_size)) - 1])
    return folds


#GenerateData_function
def generateData(data, size):
    x = np.linspace(np.amin(data[:, 0]), np.amax(
        data[:, 0]), size).reshape(-1, 1)
    y = np.linspace(np.amin(data[:, 1]), np.amax(
        data[:, 1]), size).reshape(-1, 1)
    xx, yy = np.meshgrid(x, y)
    X_2d = np.concatenate(
        [xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)
    return xx, yy, X_2d


#KDE_function
def KDE(data, mu, cov, per, h_min, h_max, h_step, sigma, k, sample_size):
    folds = kFold(len(data), k)
    np.random.shuffle(data)
    min_error = 1e9
    best_h = 0

    for h in np.arange(h_min, h_max, h_step):
        print("h: ", h)
        folds_h = []
        MSE = []
        for i, index in enumerate(folds):
            print("fold: ", i)
            xx, yy, X_2d = generateData(data[index[0]:index[1],:], sample_size)
            data_2d = data[index[1]+1:]
            N = np.size(X_2d, 0)
            d = np.size(data_2d, 1)  

            est_density = []  
            for x in X_2d:
                px = 1/N * 1/(h**d) * np.sum([kernel(dataPoint, x, h, sigma) for dataPoint in data_2d])
                est_density.append(px) 
            true_density = truePDF(X_2d,mu, cov, per)
            est_density = np.array(est_density)

            error = l2(true_density, est_density)
            MSE.append(error)
        mse = np.sum(MSE) / k
        print("h =", round(h, 2), ": l2 =", mse)
        if mse < min_error:
            min_error = mse
            best_h = h
    return best_h, min_error


data_2s = raw_data('dataset.csv')
data = data_2s[:,:-1]
print("2D_Data:", data.shape)


#Best_Value_H
for sigma in sigmas:
    print("sigma: ",sigma)
    best_h, min_error = KDE(data, mu, cov, per, 0.3, 0.9, 0.1, sigma, fold, sample_size)
    print("Best_Value_H: ", best_h)

