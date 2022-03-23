#!/usr/bin/env python
# coding: utf-8

# Generate_Dataset

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split


mu_s_1 = [[3, 6],
     [5, 4],
      [6, 6]]
cov_s_1 = [[[1.5, 0], [0, 1.5]],
            [[2, 0], [0, 2]],
           [[1, 0], [0, 1]]]


mu_s_2 =[[3, 6],
      [5, 4],
        [6, 6]]
cov_s_2 = [[[1.5, 0.1], [0.1, 1.5]],
            [[1, -0.20], [-0.20, 2]],
          [[2, -0.25], [-0.25, 1.5]]]

SIZE = 500

def generate_dataset(mu_s, cov_s, label_sampels_size):
    dataset = pd.DataFrame(data={'X1': [], 'X2': [], 'Y':[]})
    for i, mu_cov in enumerate(zip(mu_s, cov_s)):
            mu, cov = mu_cov
            x1, x2 = np.random.multivariate_normal(mu, cov, label_sampels_size).T
            temp = pd.DataFrame(data={'X1': x1, 'X2': x2, 'Y': [i]*label_sampels_size})
            dataset = pd.concat([dataset, temp],axis=0)
            return dataset

dataset1 = generate_dataset(mu_s_1, cov_s_1, SIZE)
dataset2 = generate_dataset(mu_s_2, cov_s_2, SIZE)
dataset1.to_csv('dataset1.csv', index=False)
dataset2.to_csv('dataset2.csv', index=False)


# Show_Dataset

def show_data(path):
    data = pd.read_csv(path)
    X_train, X_test, y_train, y_test = train_test_split(data[['X1','X2']],data['Y'], test_size=0.2, stratify=data['Y'] ,random_state=42)
    return X_train, X_test, y_train, y_test 


def data_set(path):
    data = pd.read_csv(path)
    return data


#Information_function

def data_info(X, y):
    n_features = X.shape[1]
    classes = np.unique(y)
    classes.sort()
    means = []
    priors = np.zeros(classes.size)
    return (n_features, classes, means, priors)


# Covariance_function

def class_covariance(X, mean):
    return (X - mean).T @ (X - mean) / X.shape[0]


# Fit_function

def fit(X_train, y_train):
    n_features, classes, means, priors = data_info(X_train, y_train)
    cov_matrices = []
    for i, y in enumerate(classes):
        priors[i] = y_train[y_train == y].size / y_train.size
        mean = np.mean(X_train[y_train == y], axis=0)
        means.append(np.asmatrix(mean))
        cov_matrices.append(class_covariance(np.asmatrix(X_train[y_train == y].values), means[i]))
    return means, priors, cov_matrices, classes


#Predicted-Y_function

def predict(X, means, priors, cov_matrices, classes):
    probs = np.asmatrix(np.zeros((X.shape[0], priors.size)))
    for i, _ in enumerate(classes):
        probs[:, i] = probability(X, means[i], priors[i], cov_matrices[i])
    probs_arg_max = np.argmax(probs, axis=1)
    return probs_arg_max


# Probability_Score_function

def probability(X, mean, prior, covariance_matrix):
    X = np.asmatrix(X.values)
    cov_matrix_det = np.linalg.det(covariance_matrix)
    cov_matrix_inv = np.linalg.pinv(covariance_matrix)
    Xm = X - mean
    Xm_covariance = (Xm @ cov_matrix_inv) @ Xm.T
    Xm_covariance_sum = Xm_covariance.sum(axis=1)
    return -0.5*Xm_covariance_sum - 0.5*np.log(cov_matrix_det) + np.log(prior)


# Accuracy_function_%

def my_accuracy(y,y_):
    correct =0 
    length = len(y)
    prediction = y_ > 0.5
    _y = y_.reshape(-1,1)
    correct = prediction ==_y
    my_accuracy = (np.sum(_y)/length)
    return my_accuracy  

