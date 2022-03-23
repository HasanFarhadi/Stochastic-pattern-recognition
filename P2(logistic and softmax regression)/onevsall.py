# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#train data
def Load_Train_Data():
    data = pd.read_csv("/content/drive/MyDrive/Depo/PatternHW/HW2/iris.data", header = None).to_numpy()
    data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(data[:,0:4], data[:,4], test_size= 0.2, stratify= data[:,4], random_state=42)
    data_train_y = np.reshape(data_train_y, ((len(data_train_y)), 1))
    data_train = np.append(data_train_x, data_train_y, axis = 1)
    return data_train

#Test data
def Load_Test_Data():
    data = pd.read_csv("/content/drive/MyDrive/Depo/PatternHW/HW2/iris.data", header = None).to_numpy()
    data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(data[:,0:4], data[:,4], test_size= 0.2, stratify= data[:,4], random_state=42)
    data_test_y = np.reshape(data_test_y, ((len(data_test_y)), 1))
    data_test = np.append(data_test_x, data_test_y, axis = 1)
    return data_test

Train_Data = Load_Train_Data()
#Train_Data

#One vs all Datasets
setosa_data = copy.deepcopy(Train_Data)

for i in range(len(setosa_data)):
    if (setosa_data[i, 4] == 'Iris-setosa'):
        setosa_data[i,4] = 1
    else:
        setosa_data[i,4] = 0

virgin_data = copy.deepcopy(Train_Data)

for i in range(len(virgin_data)):
    if (virgin_data[i, 4] == 'Iris-virginica'):
        virgin_data[i,4] = 1
    else:
        virgin_data[i,4] = 0

versi_data = copy.deepcopy(Train_Data)
for i in range(len(versi_data)):
    if (versi_data[i, 4] == 'Iris-versicolor'):
        versi_data[i,4] = 1
    else:
        versi_data[i,4] = 0

def Normal(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

def Sigmoid(x):
    sig = 1 / (1 + np.exp(-x.astype(float)))
    return sig

def LogCost(yhat, y):
    L = -np.mean(y * np.log(yhat.astype(float)) + (1 - y) * np.log((1 - yhat).astype(float)))
    return L

def GDescent(x, y , yhat, theta, iterations, learning_rate):
    model_loss = []
    for i in range(iterations):
        theta_gradient = np.dot(x.T, (yhat - y))
        theta = theta - np.dot(theta_gradient, learning_rate)
        yhat = Sigmoid(np.dot(x, theta))
        model_loss.append(LogCost(yhat, y))
    return theta, yhat, model_loss

def Accuracy(y, decision, m):
    correct = 0
    for i in range(m):
        if decision[i] == y[i,0]:
            correct +=1
    return (correct/m)*100

def Decision(yhat, m):
    decision = []
    for i in range(m):
        if yhat[i] > 0.5:
            decision.append(1)
        else:
            decision.append(0)
    decision = np.array(decision)
    return decision

#OnevAll classifier initial values and data preparation
setosa_train_x, setosa_test_x, setosa_train_y, setosa_test_y = train_test_split(setosa_data[:,0:4], setosa_data[:,4], test_size= 0.2, stratify = setosa_data[:,4])
virgin_train_x, virgin_test_x, virgin_train_y, virgin_test_y = train_test_split(virgin_data[:,0:4], virgin_data[:,4], test_size= 0.2, stratify = virgin_data[:,4])
versi_train_x, versi_test_x, versi_train_y, versi_test_y = train_test_split(versi_data[:,0:4], versi_data[:,4], test_size= 0.2, stratify = versi_data[:,4])

setosa_train_y = np.reshape(setosa_train_y, (96,1))
virgin_train_y = np.reshape(virgin_train_y, (96,1))
versi_train_y = np.reshape(versi_train_y, (96,1))

setosa_m = len(setosa_train_x)
virgin_m = len(virgin_train_x)
versi_m = len(versi_train_x)

setosa_train_x = Normal(setosa_train_x)
virgin_train_x = Normal(virgin_train_x)
versi_train_x = Normal(versi_train_x)

setosa_train_x = np.append(np.ones((96,1)), setosa_train_x, axis=1)
virgin_train_x = np.append(np.ones((96,1)), virgin_train_x, axis=1)
versi_train_x = np.append(np.ones((96,1)), versi_train_x, axis=1)

setosa_theta = np.ones((5,1))
virgin_theta = np.ones((5,1))
versi_theta = np.ones((5,1))

iterations = 10000
learning_rate = 0.01

setosa_yhat = np.zeros((setosa_m, 1))
virgin_yhat = np.zeros((virgin_m, 1))
versi_yhat = np.zeros((versi_m, 1))

#Model training
def Model(model_train_x, model_train_y, model_test_x, model_test_y, iterations, learning_rate):
    model_theta = np.ones((5,1))
    model_m = len(model_train_x)
    model_yhat = np.zeros((model_m, 1))
    model_theta, model_yhat, model_loss = GDescent(model_train_x, model_train_y, model_yhat, model_theta, iterations, learning_rate)
    model_decision = Decision(model_yhat, model_m)
    model_acc = Accuracy(model_train_y, model_decision, model_m)
    print(f"training accuracy : {model_acc} percent")

    model_test_y = np.array(model_test_y)
    model_test_x = (model_test_x - model_test_x.min()) / (model_test_x.max() - model_test_x.min())
    model_test_m = len(model_test_x)
    model_test_y = np.reshape(model_test_y, (model_test_m, 1))
    model_test_x = np.append(np.ones((model_test_m,1)), model_test_x, axis= 1)
    model_test_yhat = Sigmoid(np.dot(model_test_x, model_theta))

    model_test_decision = Decision(model_test_yhat, model_test_m)
    model_test_accuracy = Accuracy(model_test_y, model_test_decision, model_test_m)
    print(f"testing accuracy :{model_test_accuracy} percent")
    return model_theta, model_loss

print("Setosa Model")
setosa_theta, setosa_loss = Model(setosa_train_x, setosa_train_y, setosa_test_x, setosa_test_y, iterations, learning_rate)

print("Virginica Model")
virgin_theta, virgin_loss = Model(virgin_train_x, virgin_train_y, virgin_test_x, virgin_test_y, iterations, learning_rate)

print("Versicolor Model")
versi_theta, versi_loss = Model(versi_train_x, versi_train_y, versi_test_x, versi_test_y, iterations, learning_rate)

#loading our initial 20% unused data
data_test = Load_Test_Data()

for i in range(30):
        if (data_test[i, 4] == 'Iris-versicolor'):
            data_test[i,4] = 1
        elif (data_test[i,4] == 'Iris-virginica'):
            data_test[i,4] = 2
        else:
            data_test[i,4] = 0

data_test_x = data_test[:, 0:4]
data_test_y = data_test[: , 4]

data_test_x = np.append(np.ones((30, 1)), data_test_x, axis=1)
data_test_y = np.reshape(data_test_y, (30,1))

setosa_weights = Sigmoid(np.dot(data_test_x, setosa_theta))
virgin_weights = Sigmoid(np.dot(data_test_x, virgin_theta))
versi_weights = Sigmoid(np.dot(data_test_x, versi_theta))

weights = setosa_weights
weights = np.append(setosa_weights, versi_weights, axis = 1)
weights = np.append(weights, virgin_weights, axis = 1)

OneVsAll_prediction = np.argmax(weights, axis = 1)
#reporting the output
final_accuracy = Accuracy(data_test_y, OneVsAll_prediction, 30)
print(f"The Accuarcy of the One Versus All method is {final_accuracy} percent")

plt.figure(1)
plt.plot(setosa_loss)
plt.title("Setosa model loss in each step")
plt.figure(2)
plt.plot(versi_loss)
plt.title("Versicolor model loss in each step")
plt.figure(3)
plt.plot(virgin_loss)
plt.title("Virginica model loss in each step")