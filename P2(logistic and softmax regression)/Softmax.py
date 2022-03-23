#!/usr/bin/env python
# coding: utf-8

# # Show_DataSet

# In[30]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[31]:


cal_names=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Classes']
Data = pd.read_csv('iris.data', sep="," , names= cal_names)
Data


# In[32]:


Class = Data['Classes']
Class.replace({"Iris-setosa": 0., "Iris-virginica": 1., "Iris-versicolor": 2.}, inplace=True)
Data


# # Split_DataSet_Train_Test

# In[33]:


def my_random_data() :
    X_train, X_test, y_train, y_test = train_test_split(Data.iloc[ : , 0:-1], Data.iloc[ : , -1], stratify=Data['Classes'],test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# In[34]:


my_random_data()


# # Add_Bias_to_Data_function

# In[35]:


def my_bias_to_data(x):
    x = np.concatenate((np.ones((len(x))).reshape(-1,1), x),axis = 1)
    return x 


# # One_Hot_Encoding

# In[36]:


def one_hot(y):
    data = list(set(y))
    re_represent_data = np.zeros(len(y)*len(data)).reshape(len(y),len(data))
    for i in range(len(data)):
        re_represent_data[y==data[i],i] = 1
    return re_represent_data


# In[37]:


one_hot(Data.iloc[ : , -1])


# # Usable_Data

# In[38]:


def usable_data() :
    X_train, X_test, y_train, y_test = my_random_data()
    X_train, X_test = my_bias_to_data(X_train), my_bias_to_data(X_test)
    y_train, y_test = one_hot(y_train), one_hot(y_test)
    return  X_train, X_test, y_train, y_test 


# In[39]:


X_train, X_test, y_train, y_test = usable_data()


# In[40]:


usable_data()


# # Softmax_function

# In[41]:


def softmax(z):
    z -= np.max(z)
    a = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return a


# # Gradient_descent_function

# In[42]:


def gradient_dsc(X, y, y_):
    return np.dot(X.T, (y_ - y)) / y.size


# # SoftmaxRegression_Predicted-Y_function

# In[43]:


def predicted_y_Soft (x, theta) :
    z = np.dot(x, theta.T)
    return softmax(z)


# # Loss_function

# In[44]:


def loss(y_, y):
    return -1/y.size * np.sum(y * np.log(y_) + (1 - y) * np.log(1 - y_), axis=0)


# In[45]:


loss_log = []


# # Thetas

# # Thetas_First_function

# In[46]:


def Theta_first(shape):
    return np.zeros(shape, dtype='float64')


# In[47]:


Thetas = Theta_first((y_train.shape[1], X_train.shape[1]))
Thetas


# # Train_function

# In[48]:


def train(X_train, y_train, learnRate, iteration, Thetas): 
   for i in range(iteration): 
       XTheta = np.dot(X_train, Thetas.T)
       probability = softmax(XTheta)
       grad = gradient_dsc(X_train, y_train, probability)
       Thetas =Thetas - learnRate * grad.T
       my_loss = loss(probability, y_train)
       loss_log.append(my_loss)
   return Thetas

iteration = 100000
learnRate = 0.01


# In[49]:


Thetas = train (X_train, y_train, learnRate, iteration , Thetas)
Thetas


# # Predicted-Y_function

# In[50]:


def predicted_y(X, Thetas):
    a = predicted_y_Soft (X, Thetas)
    y_ = np.argmax(a, axis=1)
    return y_


# # Predicted-Y_Train

# In[51]:


predicted_y(X_train, Thetas)


# In[52]:


predicted_y_Train = predicted_y(X_train, Thetas)


# # Predicted-Y_Test

# In[53]:


predicted_y(X_test, Thetas)


# In[54]:


predicted_y_Test = predicted_y(X_test, Thetas)


# # Accuracy_function

# In[55]:


def my_accuracy(y,y_):
    correct =0 
    length = len(y)
    prediction = y_ > 0.5
    _y = y_.reshape(-1,1)
    correct = prediction ==_y
    my_accuracy = (np.sum(_y)/length)*100
    return my_accuracy


# # Accuracy_Train_%

# In[56]:


my_accuracy(y_train, predicted_y_Train)


# # Accuracy_Test_%

# In[57]:


my_accuracy(y_test, predicted_y_Test)

