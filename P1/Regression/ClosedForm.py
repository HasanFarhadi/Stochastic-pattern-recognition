# -*- coding: utf-8 -*-

#................Show_Dataset..............#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data_Train
Train = pd.read_csv('Data-Train.csv')
Train

#Data_Test
Test = pd.read_csv('Data-Test.csv')
Test


#...............Data_Normalization................#

x_data_train = Train [['x']]
y_data_train = Train [['y']]
x_data_test = Test [['x']]
y_data_test = Test [['y']]

x_norm_data_train = x_data_train.apply(lambda x_data_train: (x_data_train - x_data_train.min(axis=0)) / (x_data_train.max(axis=0) - x_data_train.min(axis=0)) )
x_norm_data_test = x_data_test.apply(lambda x_data_test: (x_data_test - x_data_test.min(axis=0)) / (x_data_test.max(axis=0) - x_data_test.min(axis=0)) )

#Show_Normalized_X_Data_Train
x_norm_data_train

#Show_Normalized_X_Data_Test
x_norm_data_test



#...............LinearRegression_using_ClosedForm...............#

#Calculate_mean_function
def cal_mean(data) :
    data_total = np.sum(data).to_numpy(dtype='float64') 
    mean = data_total / len(data)
    return mean

#LinearRegression_using_ClosedForm_for_Data_Train#

#Theta1_Train
theta1_Train = np.linalg.inv(x_norm_data_train.T.dot(x_norm_data_train)).dot(x_norm_data_train.T).dot(y_data_train)
theta1_Train


#Theta0_Train
theta0_Train = cal_mean(y_data_train) - (theta1_Train * cal_mean(x_norm_data_train))
theta0_Train


#LinearRegression_Predicted-Y_function
def predicted_y (x, y) :
    y_LinearRegression = np.dot(y.T, x.T).flatten()
    return y_LinearRegression  


#Predicted-Y_LinearRegression_Train
y_LinearRegression_train = predicted_y (x_norm_data_train, theta1_Train)


#Plot_DataTrain
plt.scatter(x_norm_data_train, y_data_train, color='lime')
plt.title('DataTrain & LinearRegression_ClosedForm')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_norm_data_train, y_LinearRegression_train, color='blue' )
plt.legend (['LinearRegression', 'Data_Train'], loc='best')


#Predicted-Y_Train
y_predicted_train = x_norm_data_train.dot(theta1_Train) 


#MSE_function
def my_MSE (y, y_):
    y= y.to_numpy(dtype='float64')
    y_= y_.to_numpy(dtype='float64')
    n=len(y)
    for i in range (0,n) :
        difference = np.subtract(y_, y)
        A = np.power(difference , 2) 
        return np.sum(A) / n


#MSE_Train
my_MSE (y_data_train, y_predicted_train)


#LinearRegression_using_ClosedForm_for_Data_Test#

#Theta1_Test
theta1_Test = np.linalg.inv(x_norm_data_test.T.dot(x_norm_data_test)).dot(x_norm_data_test.T).dot(y_data_test)
theta1_Test


#Theta0_Test
theta0_Test = cal_mean(y_data_test) - (theta1_Test * cal_mean(x_norm_data_test))
theta0_Test


#Predicted-Y_LinearRegression_Test
y_LinearRegression_test = predicted_y (x_norm_data_test, theta1_Test)


#Plot_DataTest
plt.scatter(x_norm_data_test, y_data_test, color='cyan')
plt.title('DataTest & LinearRegression_ClosedForm')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_norm_data_test, y_LinearRegression_test, color='navy')
plt.legend (['LinearRegression', 'Data_Test'], loc='best')


#Predicted-Y_Test
y_predicted_test = x_norm_data_test.dot(theta1_Test)


#MSE_Test
my_MSE (y_data_test, y_predicted_test)

