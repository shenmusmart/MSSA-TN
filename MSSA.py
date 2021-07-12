# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
from sklearn.svm import SVC
from sklearn import metrics
import h5py
from sklearn import preprocessing
from scipy.linalg import fractional_matrix_power
from scipy.io import savemat

# load two different subject data, one for training, the other for testing
# data_structure: N*360*120  (N sample number)*(3 second *
# 120Hz sampling rate)*(30 channels *4 spectral bands)

path = 'data.mat'
matfile = h5py.File(path)
print(matfile.keys())
print(matfile.values())
data = matfile['data']

train_data=data['train_data'][:]
train_label=data['train_label'][:]
train_label=np.squeeze(train_label)
test_data=data['test_data'][:]
test_label=data['test_label'][:]
test_label=np.squeeze(test_label)

print(np.shape(train_data))
print(np.shape(train_label))
print(np.shape(test_data))
print(np.shape(test_label))

train_feature=np.zeros((np.shape(train_data)[0],120))
test_feature=np.zeros((np.shape(test_data)[0],120))

for i in range(np.shape(train_data)[0]):
    for j in range(120):
        train_feature[i,j]=np.var(train_data[i,:,j],ddof=1)
    
for i in range(np.shape(test_data)[0]):
    for j in range(120):
        test_feature[i,j]=np.var(test_data[i,:,j],ddof=1)
    

train_x=train_feature
train_y=train_label
test_x=test_feature
test_y=test_label

# choose whether the data is noamlized
# train_x = preprocessing.StandardScaler().fit_transform(train_x)
# test_x = preprocessing.StandardScaler().fit_transform(test_x)


model = SVC(kernel='linear', C=1.0)
model.fit(train_x,train_y)
pred = model.predict(test_x)
accuracy = metrics.accuracy_score(test_y, pred)*100
print('Classification results without MSSA')
print(accuracy)

# MSSA(Multi-Source Signal Alignment)

# calculate covariance in each spectral band for train_data and test_data

cov_train_data=np.zeros((4,30,30,np.shape(train_data)[0]))
cov_test_data=np.zeros((4,30,30,np.shape(test_data)[0]))
print(np.shape(cov_train_data))
for i in range(np.shape(train_data)[0]):
    cov_train_data[0,:,:,i]=np.matmul(train_data[i,:,0:30].T,train_data[i,:,0:30])
    cov_train_data[1,:,:,i]=np.matmul(train_data[i,:,30:60].T,train_data[i,:,30:60])
    cov_train_data[2,:,:,i]=np.matmul(train_data[i,:,60:90].T,train_data[i,:,60:90])
    cov_train_data[3,:,:,i]=np.matmul(train_data[i,:,90:120].T,train_data[i,:,90:120])
                       
for i in range(np.shape(test_data)[0]):
    cov_test_data[0,:,:,i]=np.matmul(test_data[i,:,0:30].T,test_data[i,:,0:30])
    cov_test_data[1,:,:,i]=np.matmul(test_data[i,:,30:60].T,test_data[i,:,30:60])
    cov_test_data[2,:,:,i]=np.matmul(test_data[i,:,60:90].T,test_data[i,:,60:90])
    cov_test_data[3,:,:,i]=np.matmul(test_data[i,:,90:120].T,test_data[i,:,90:120])


# calculate transformation matrix A

# regularization parameter lamda
lamda=1

cov_train_data_band_1=np.sum(np.squeeze(cov_train_data[0,:,:,:]),2)/(360*np.shape(train_data)[0]-1)+ lamda*np.identity(30)
cov_train_data_band_2=np.sum(np.squeeze(cov_train_data[1,:,:,:]),2)/(360*np.shape(train_data)[0]-1)+ lamda*np.identity(30)
cov_train_data_band_3=np.sum(np.squeeze(cov_train_data[2,:,:,:]),2)/(360*np.shape(train_data)[0]-1)+ lamda*np.identity(30)
cov_train_data_band_4=np.sum(np.squeeze(cov_train_data[3,:,:,:]),2)/(360*np.shape(train_data)[0]-1)+ lamda*np.identity(30)
                    
                        

cov_test_data_band_1=np.sum(np.squeeze(cov_test_data[0,:,:,:]),2)/(360*np.shape(test_data)[0]-1)+ lamda*np.identity(30)
cov_test_data_band_2=np.sum(np.squeeze(cov_test_data[1,:,:,:]),2)/(360*np.shape(test_data)[0]-1)+ lamda*np.identity(30)
cov_test_data_band_3=np.sum(np.squeeze(cov_test_data[2,:,:,:]),2)/(360*np.shape(test_data)[0]-1)+ lamda*np.identity(30)
cov_test_data_band_4=np.sum(np.squeeze(cov_test_data[3,:,:,:]),2)/(360*np.shape(test_data)[0]-1)+ lamda*np.identity(30)


A_1 = np.matmul(fractional_matrix_power(cov_test_data_band_1,1/2),fractional_matrix_power(cov_train_data_band_1,-1/2))
A_2 = np.matmul(fractional_matrix_power(cov_test_data_band_2,1/2),fractional_matrix_power(cov_train_data_band_2,-1/2))
A_3 = np.matmul(fractional_matrix_power(cov_test_data_band_3,1/2),fractional_matrix_power(cov_train_data_band_3,-1/2))
A_4 = np.matmul(fractional_matrix_power(cov_test_data_band_4,1/2),fractional_matrix_power(cov_train_data_band_4,-1/2))


# Calculate transformed training data in each spectral band
train_data_transfer=np.zeros((120,360,np.shape(train_data)[0]));
for k in range(np.shape(train_data)[0]):
    train_data_transfer[0:30,:,k]=np.matmul(A_1,train_data[k,:,0:30].T)
    train_data_transfer[30:60,:,k]=np.matmul(A_2,train_data[k,:,30:60].T)
    train_data_transfer[60:90,:,k]=np.matmul(A_3,train_data[k,:,60:90].T)
    train_data_transfer[90:120,:,k]=np.matmul(A_4,train_data[k,:,90:120].T)
    
# extract feature and classify after MSSA

feature_transfer_train_data=np.zeros((np.shape(train_data)[0],120))
test_feature=np.zeros((np.shape(test_data)[0],120))


for i in range(np.shape(train_data)[0]):
    for j in range(120):
        feature_transfer_train_data[i,j]=np.var(train_data_transfer[j,:,i],ddof=1)
for i in range(np.shape(test_data)[0]):
    for j in range(120):
        test_feature[i,j]=np.var(test_data[i,:,j],ddof=1)
    
print(np.shape(feature_transfer_train_data))
print(np.shape(test_data))

train_x=feature_transfer_train_data
train_y=train_label
test_x=test_feature
test_y=test_label

# train_x = preprocessing.StandardScaler().fit_transform(train_x)
# test_x = preprocessing.StandardScaler().fit_transform(test_x)

model = SVC(kernel='linear', C=1.0)
model.fit(train_x,train_y)
pred = model.predict(test_x)
accuracy = metrics.accuracy_score(test_y, pred)*100
print('Classification results after MSSA')
print(accuracy)

#Calculate multi-dimensional features for tensor network classification

tensor_train_data=np.zeros((4,30,30,np.shape(train_data)[0]))
tensor_test_data=np.zeros((4,30,30,np.shape(test_data)[0]))

for i in range(np.shape(train_data)[0]):
    tensor_train_data[0,:,:,i]=np.matmul(train_data_transfer[0:30,:,i],train_data_transfer[0:30,:,i].T)
    tensor_train_data[1,:,:,i]=np.matmul(train_data_transfer[30:60,:,i],train_data_transfer[30:60,:,i].T)
    tensor_train_data[2,:,:,i]=np.matmul(train_data_transfer[60:90,:,i],train_data_transfer[60:90,:,i].T)
    tensor_train_data[3,:,:,i]=np.matmul(train_data_transfer[90:120,:,i],train_data_transfer[90:120,:,i].T)
    
for i in range(np.shape(test_data)[0]):
    tensor_test_data[0,:,:,i]=np.matmul(test_data[i,:,0:30].T,test_data[i,:,0:30])
    tensor_test_data[1,:,:,i]=np.matmul(test_data[i,:,30:60].T,test_data[i,:,30:60])
    tensor_test_data[2,:,:,i]=np.matmul(test_data[i,:,60:90].T,test_data[i,:,60:90])
    tensor_test_data[3,:,:,i]=np.matmul(test_data[i,:,90:120].T,test_data[i,:,90:120])
    
    

# Reshape to vactor form
tensor_train_data=np.reshape(tensor_train_data,(4*30*30,np.shape(train_data)[0]),order='F')
tensor_test_data=np.reshape(tensor_test_data,(4*30*30,np.shape(test_data)[0]),order='F')

dic = {"train_data": tensor_train_data,"train_label":train_label, "test_data":tensor_test_data,"test_label":test_label}
savemat("tensor_feature.mat", dic)
