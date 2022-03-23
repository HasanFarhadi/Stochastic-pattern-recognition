# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
train = pd.read_csv("/content/drive/MyDrive/Depo/Data-Train.csv")
 
#separating the training data into x and y axes
x = np.matrix(train.iloc[:, 0])
y = np.matrix(train.iloc[:, 1])

#Normalizing The Input Data 
x = x / 100

#reshaping x and y to column matrices
x = np.reshape(x, (1000, 1))
y = np.reshape(y, (1000,1))
#assigning initial values to model parameters
randomx = np.zeros((1,2))
randomy = np.zeros((1,1))
randomyhat = np.zeros((1,1))
n =len(x)
yhat = np.zeros((1000,1))
x = np.append(np.ones((n,1)),x,axis=1) 
iteration_error = []

theta = np.zeros((2,1))
theta_gradient = np.ones((2,1))

# Calculating Squared sum of error over x(a) , y(b) and ŷ(bhat) using theta(c) over n points
def MSE(a, b, c, n):
    bhat = np.dot(a, c)
    SquaredError = np.power(bhat - b, 2)
    error = (1/(2*n))*(np.sum(SquaredError))
    return error

error = MSE(x, y, theta, n)

# stochastic gradient descent which trains our model based on one randomly picked training data in each iteration
def GDescent(x, y, theta, learning_rate):
    #picking a random number in input 
    randnum = np.random.randint(0, n, 1)
    randomx = x[randnum, :]
    randomy = y[randnum, :]
    randomyhat = np.dot(randomx, theta)
    #x is a 1*2 matrix and ŷ-y is a 1*1 matrix so dot product of (ŷ-y , x) which yields Σ(hΘ(X1)-y1)X1 for our random number
    #would result in our theta matrix to be a row matrix so instead we can calculate the dot product of (xT , ŷ-y) which bears the same values but
    #results in theta being a column vector  
    theta_gradient = np.dot(randomx.T,(randomyhat-randomy))
    theta -= theta_gradient * learning_rate
    return theta

def display(x, y,yhat):
    plt.figure(figsize =(3,3), dpi = 500)
    plt.figure(1)
    plt.scatter([x], [y], s = 0.2, c = "red")
    plt.plot(x, yhat, c = "black")
    plt.xlabel("features")
    plt.ylabel("label")

def displaycost(iterations, iteration_error):
    plt.figure(2)
    plt.figure(figsize =(3,3), dpi = 500)
    plt.plot(iteration_error, c = "blue" )
    plt.xlabel("number of iterations")
    plt.ylabel("error")

learning_rate = 0.001
iterations = 1 
#training our model until we achieve a MSE lower than 4.5 or we surpass 100k iterations
while ((MSE(x, y, theta, n) > 4.5) & (iterations < 1e5)):
    theta = GDescent(x, y, theta, learning_rate)
    #we calculate squared sum of error of all data after every iteration and append it to iteration_error list
    iteration_error.append(MSE(x, y, theta, n))
    iterations += 1
#we update ŷ for all inputs
train_error = MSE(x , y, theta, n)
print(f"Mean Squared Error over all of the training data is : {train_error}")
yhat = np.dot(x,theta)
display(x[:, 1],y,yhat)
displaycost(iterations, iteration_error)

#loading the test data
test = pd.read_csv("/content/drive/MyDrive/Depo/Data-Test.csv")
test_x = np.matrix(test.iloc[:, 0])
test_y = np.matrix(test.iloc[:, 1])
test_x = test_x / 100
test_x = test_x.T
test_y = test_y.T
test_n = len(test_x)
test_x = np.append(np.ones((test_n, 1)), test_x, axis=1)

#predicting the output of test data 
test_yhat = np.dot(test_x, theta)
test_error  = MSE(test_x, test_y, theta, test_n)
print(f"Mean Squared Error over the test data is : {test_error}")

display(test_x[:, 1], test_y, test_yhat)

theta_gradient