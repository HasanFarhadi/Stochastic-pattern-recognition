#!/usr/bin/env python
# coding: utf-8

#Show_Dataset
import cv2
print(cv2.__version__)


import pandas as pd
import numpy  as np
from numpy import linalg as LA
import os
from string import digits
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import imshow

folder = 'jaffe'
images_path = os.listdir(folder)
feelings = []
for i, p in enumerate(images_path):
    images_path[i] = os.path.join(folder, p)
    remove_digits = str.maketrans('', '', digits)
    feeling = p.split('.')[1].translate(remove_digits)
    feelings.append(feeling)


#Split_Data
data = pd.DataFrame({'path': images_path, 'feeling': feelings})
fear = data[data['feeling'].isin(['FE'])]
happy = data[data['feeling'].isin(['HA'])]
sad = data[data['feeling'].isin(['SA'])]
neutral = data[data['feeling'].isin(['NE'])]
surprised = data[data['feeling'].isin(['SU'])]
data.sample(5)


fe = []
ha = []
sa = []
ne = []
su = []
for i, row in data.iterrows():    
    im = Image.open(row['path']).convert('L').resize((64,64))
    im = np.array(im)
    feeling = row['feeling']
    if feeling == 'FE':
        fe.append(im)
    elif (feeling == 'HA'):
        ha.append(im)
    elif (feeling == 'SA'):
        sa.append(im)
    elif (feeling == 'NE'):
        ne.append(im)
    elif (feeling == 'SU'):
        su.append(im)

fe = np.array(fe)
ha = np.array(ha)
sa = np.array(sa)
ne = np.array(ne)
su = np.array(su)

all_images = np.concatenate((fe, ha, sa, ne, su))
all_images.shape


#Visualize_the_Dataset
images = all_images
images = images.transpose(1,2,0)
images = images/255
print(images.shape)
fig = plt.figure(figsize=(15,4))
for i in range(16):
    plt.subplot(2,8,i+1)
    plt.imshow(images[:,:,i],cmap = 'gray')
    plt.axis('off')


#Preprocess_and_Normalize
mean = np.mean(images, axis = 2)
plt.figure(figsize=(4,4))
plt.imshow(mean, cmap = 'gray')
plt.axis('off')

demeaned = images - mean[...,None]
fig = plt.figure(figsize=(15,4))
for i in range(16):
    plt.subplot(2,8,i+1)
    plt.imshow(demeaned[:,:,i],cmap = 'gray')
    plt.axis('off')


#EigenFaces
A = np.zeros((N*N, M))
for i in range(M):
    A[:,i] = demeaned[:,:,i].flatten()
L = np.matmul(A.transpose(), A)
mu, v = LA.eig(L) 
idx = mu.argsort()[::-1]
mu = mu[idx]
v = v[:,idx]
u = np.matmul(A, v) 


#Visualize_2D_and_3D_plots
M = images.shape[2]
N = images.shape[0] 

plt.rcParams["figure.figsize"] = (10, 8)
flatten_images = np.zeros((N*N, M))
for i in range(M):
    flatten_images[:,i] = images[:,:,i].flatten()
num_components = 2
components = u[:, :num_components]
proj_data = (flatten_images.T @ components)
plt.scatter(proj_data[:, 0], proj_data[:,1], marker='o', color='gold')
plt.show()


num_components = 3
components = u[:, :num_components]
proj_data = (flatten_images.T @ components)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(proj_data[:, 0], proj_data[:,1], proj_data[:,2], marker='o', color='gold')
plt.show()


#Visualize_some_first_principal_components
eigfaces = np.zeros((N,N,M))
for i in range(M):
    eigfaces[:,:,i] = u[:,i].reshape(N,N)
fig = plt.figure(figsize=(10, 10))
for i in range(24):
    plt.subplot(6,6,i+1)
    plt.imshow(eigfaces[:,:,i],cmap = 'gray')
    plt.axis('off')


#Reconstructed_images_K=1
flatten_images = np.zeros((N*N, M))
for i in range(M):
    flatten_images[:,i] = images[:,:,i].flatten()
num_components = 1
components = u[:, :num_components]
projected_data = flatten_images.T @ components
reconstructed_data = (projected_data @ components.T).T
new_images = np.zeros((N,N,M))
for i in range(M):
    new_images[:,:,i] = reconstructed_data[:,i].reshape(N,N)
fig = plt.figure(figsize=(10, 10))
for i in range(24):
    plt.subplot(6,6,i+1)
    plt.imshow(eigfaces[:,:,i],cmap = 'gray')
    plt.axis('off')


#Reconstructed_images_K=40
flatten_images = np.zeros((N*N, M))
for i in range(M):
    flatten_images[:,i] = images[:,:,i].flatten()
num_components = 40
components = u[:, :num_components]
projected_data = flatten_images.T @ components
reconstructed_data = (projected_data @ components.T).T
new_images = np.zeros((N,N,M))
for i in range(M):
    new_images[:,:,i] = reconstructed_data[:,i].reshape(N,N)
fig = plt.figure(figsize=(10, 10))
for i in range(24):
    plt.subplot(6,6,i+1)
    plt.imshow(eigfaces[:,:,i],cmap = 'gray')
    plt.axis('off')


#Reconstructed_images_K=120
flatten_images = np.zeros((N*N, M))
for i in range(M):
    flatten_images[:,i] = images[:,:,i].flatten()
num_components = 120
components = u[:, :num_components]
projected_data = flatten_images.T @ components
reconstructed_data = (projected_data @ components.T).T
new_images = np.zeros((N,N,M))
for i in range(M):
    new_images[:,:,i] = reconstructed_data[:,i].reshape(N,N)
fig = plt.figure(figsize=(10, 10))
for i in range(24):
    plt.subplot(6,6,i+1)
    plt.imshow(eigfaces[:,:,i],cmap = 'gray')
    plt.axis('off')


#MSE_K=1
from sklearn.metrics import mean_squared_error
mse = []
for i in range(flatten_images.shape[1]):
    mse.append(mean_squared_error(flatten_images[:, i], reconstructed_data[:, i]))
plt.scatter(list(range(len(mse))), mse, marker='o', color='orangered')
plt.suptitle('MSE between the original and reconstructed images for K=1')
plt.show()  


#MSE_K=40
mse = []
for i in range(flatten_images.shape[1]):
    mse.append(mean_squared_error(flatten_images[:, i], reconstructed_data[:, i]))
plt.scatter(list(range(len(mse))), mse, marker='o', color='limegreen')
plt.suptitle('MSE between the original and reconstructed images for K=40')
plt.show()  


#MSE_K=120
mse = []
for i in range(flatten_images.shape[1]):
    mse.append(mean_squared_error(flatten_images[:, i], reconstructed_data[:, i]))
plt.scatter(list(range(len(mse))), mse, marker='o', color='magenta')
plt.suptitle('MSE between the original and reconstructed images for K=120')
plt.show()  


#Cumulative_Variance_PCA
total_var = sum(mu)
cum_var = np.cumsum(mu)/total_var
num_comp = range(1,len(mu)+1)
plt.title
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Variance')
plt.scatter(num_comp, cum_var)
plt.show()

