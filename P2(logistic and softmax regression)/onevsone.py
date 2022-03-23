# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#loading test and train data
def Load_Train_Data():
    data = pd.read_csv("/content/drive/MyDrive/Depo/PatternHW/HW2/iris.data", header = None).to_numpy()
    data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(data[:,0:4], data[:,4], test_size= 0.2, stratify= data[:,4], random_state=42)
    data_train_y = np.reshape(data_train_y, ((len(data_train_y)), 1))
    data_train = np.append(data_train_x, data_train_y, axis = 1)
    return data_train

def Load_Test_Data():
    data = pd.read_csv("/content/drive/MyDrive/Depo/PatternHW/HW2/iris.data", header = None).to_numpy()
    data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(data[:,0:4], data[:,4], test_size= 0.2, stratify= data[:,4], random_state=42)
    data_test_y = np.reshape(data_test_y, ((len(data_test_y)), 1))
    data_test = np.append(data_test_x, data_test_y, axis = 1)
    return data_test

Train_Data = Load_Train_Data()

#modifing the inout data for 1v1 classifiers
#setvers train data
setvers_data = copy.deepcopy(Train_Data)
setvers_data = np.delete(setvers_data, np.where(setvers_data == 'Iris-virginica'), 0)
for i in range(len(setvers_data)):
    if (setvers_data[i, 4] == 'Iris-setosa'):
        setvers_data[i,4] = 1
    else:
        setvers_data[i,4] = 0

setvir_data = copy.deepcopy(Train_Data)
setvir_data = np.delete(setvir_data, np.where(setvir_data == 'Iris-versicolor'), 0)
for i in range(len(setvir_data)):
    if (setvir_data[i, 4] == 'Iris-setosa'):
        setvir_data[i,4] = 1
    else:
        setvir_data[i,4] = 0

virvers_data = copy.deepcopy(Train_Data)
virvers_data = np.delete(virvers_data, np.where(virvers_data == 'Iris-setosa'), 0)
for i in range(len(virvers_data)):
    if (virvers_data[i, 4] == 'Iris-virginica'):
        virvers_data[i,4] = 1
    else:
        virvers_data[i,4] = 0

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

setvers_train_x, setvers_test_x, setvers_train_y, setvers_test_y = train_test_split(setvers_data[:,0:4], setvers_data[:,4], test_size= 0.2)
setvir_train_x, setvir_test_x, setvir_train_y, setvir_test_y = train_test_split(setvir_data[:,0:4], setvir_data[:,4], test_size= 0.2)
virvers_train_x, virvers_test_x, virvers_train_y, virvers_test_y = train_test_split(virvers_data[:,0:4], virvers_data[:,4], test_size= 0.2)

setvers_train_y = np.reshape(setvers_train_y, (63,1))
setvir_train_y = np.reshape(setvir_train_y, (64,1))
virvers_train_y = np.reshape(virvers_train_y, (63,1))


setvers_m = len(setvers_train_x)
setvir_m = len(setvir_train_x)
virvers_m = len(virvers_train_x)



setvers_train_x = Normal(setvers_train_x)
setvir_train_x = Normal(setvir_train_x)
virvers_train_x = Normal(virvers_train_x)


setvers_train_x = np.append(np.ones((63,1)), setvers_train_x, axis=1)
setvir_train_x = np.append(np.ones((64,1)), setvir_train_x, axis=1)
virvers_train_x = np.append(np.ones((63,1)), virvers_train_x, axis=1)


setvers_theta = np.ones((5,1))
setvir_theta = np.ones((5,1))
virvers_theta = np.ones((5,1))

#np.shape(virvers_train_y)

iterations = 10000
learning_rate = 0.01

setvers_yhat = np.zeros((setvers_m, 1))
setvir_yhat = np.zeros((setvir_m, 1))
virvers_yhat = np.zeros((virvers_m, 1))

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

print("Setosa/Versicolor Model")
setvers_theta, setvers_loss = Model(setvers_train_x, setvers_train_y, setvers_test_x, setvers_test_y, iterations, learning_rate)

print("Setosa/Virginica Model")
setvir_theta, setvir_loss = Model(setvir_train_x, setvir_train_y, setvir_test_x, setvir_test_y, iterations, learning_rate)

print("Virginica/Versicolor Model")
virvers_theta, virvers_loss = Model(virvers_train_x, virvers_train_y, virvers_test_x, virvers_test_y, iterations, learning_rate)

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

#Evaluating the test data
setvers_weights = Sigmoid(np.dot(data_test_x, setvers_theta))
setvir_weights = Sigmoid(np.dot(data_test_x, setvir_theta))
virvers_weights = Sigmoid(np.dot(data_test_x, virvers_theta))

np.shape(setvers_weights)

setvers_result = Decision(setvers_weights, 30)
setvir_result = Decision(setvir_weights, 30)
virvers_result = Decision(virvers_weights, 30)



OneVsOne_prediction = []
for i in range(30):
    if (setvers_result[i] == 1) & (setvir_result[i] == 1):
        OneVsOne_prediction.append(0)
    elif (setvers_result[i] == 0) & (virvers_result[i] == 0):
        OneVsOne_prediction.append(1)
    elif (setvir_result[i] == 0) & (virvers_result[i] == 1):
        OneVsOne_prediction.append(2)
    else:
        OneVsOne_prediction.append(np.random.randint(0, 3))

OneVsOne_prediction = np.reshape(OneVsOne_prediction, (30,1))
final_accuracy = Accuracy(data_test_y, OneVsOne_prediction, 30)

print(f"The Accuarcy of the One Versus One method is {final_accuracy} percent")


plt.figure(1)
plt.plot(setvers_loss)
plt.title("Setosa/Versicolor model loss in each step")
plt.figure(2)
plt.plot(setvir_loss)
plt.title("Setosa/Virginica model loss in each step")
plt.figure(3)
plt.plot(virvers_loss)
plt.title("Virginica/Versicolor model loss in each step")