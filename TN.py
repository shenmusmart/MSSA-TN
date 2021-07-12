#!/usr/bin/env python
# coding: utf-8

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import scipy.io
import tensorflow as tf
import os
# Calculation with CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensornetwork as tn
# Set the backend to tesorflow
tn.set_default_backend("tensorflow")


# define tensor network parameters. Input dimension 1,2,3 are 4,30,30,
# which is corresponding to multi-dimensional features, output dimension is 2,
# bond dimension is 5
output_dimension=2
connect_dimension=5
input_dimension_1=4
input_dimension_2=30
input_dimension_3=30


class TNLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TNLayer, self).__init__()
        #Decompose weight tensor to interconnected weight matrices
        self.a_var = tf.Variable(tf.random.normal(
                shape=(input_dimension_1,connect_dimension), stddev=1.0/16.0),
                 name="a", trainable=True)
        self.b_var = tf.Variable(tf.random.normal(
                shape=(connect_dimension,input_dimension_2,connect_dimension), stddev=1.0/16.0),
                 name="b", trainable=True)
        self.c_var = tf.Variable(tf.random.normal(
                shape=(connect_dimension,output_dimension,connect_dimension), stddev=1.0/16.0),
                 name="c", trainable=True)
        self.d_var = tf.Variable(tf.random.normal(
                shape=(connect_dimension,input_dimension_3), stddev=1.0/16.0),
                 name="d", trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=(output_dimension)), name="bias", trainable=True)
        
    def call(self, inputs):
        # Reshape vector form features into 3-order tensor. 
        # contraction between feature tensors and weight matrics
        def f(input_vec, a_var,b_var,c_var,d_var,bias_var):
            print(tf.shape(input_vec))
            input_vec = tf.reshape(input_vec, (input_dimension_1,input_dimension_2,input_dimension_3))
            x_node=tn.Node(input_vec)
            a = tn.Node(a_var)
            b = tn.Node(b_var)
            c = tn.Node(c_var)
            d = tn.Node(d_var) 
            a[0] ^ x_node[0]
            a[1] ^ b[0]
            b[1] ^ x_node[1]
            b[2] ^ c[0]
            d[1] ^ x_node[2]
            c[2] ^ d[0]

            c_1 = a @ x_node
            c_2 = c_1 @ b
            c_3 = c_2 @ d
            c_4 = c_3 @ c
            
            result = c_4.tensor

            return result + bias_var
            

        result = tf.vectorized_map(
            lambda vec: f(vec, self.a_var,self.b_var,self.c_var,self.d_var,self.bias), inputs)
        return tf.nn.swish(tf.reshape(result, (-1, output_dimension)))

    
    
    


# load data
path = 'tensor_feature.mat'
data = scipy.io.loadmat(path)
print(data.keys())
x_train=data['train_data'].T
y_train=data['train_label']
y_train=np.squeeze(y_train)
x_test=data['test_data'].T
y_test=data['test_label']
y_test=np.squeeze(y_test)
# x_train,y_train=shuffle(x_train,y_train)
print(np.shape(y_train))
print(np.shape(y_test))
print(np.shape(x_train))
print(np.shape(x_test))

# vector normalization
x_train = preprocessing.StandardScaler().fit_transform(x_train)
x_test = preprocessing.StandardScaler().fit_transform(x_test)

# one-hot encoding
def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded

encoded_train_label= encode(y_train)
encoded_test_label = encode(y_test)

# define the network and optimization parameters
inputA = tf.keras.Input(shape=(4*30*30,))
Dense = tf.keras.layers.Dense
x = TNLayer()(inputA)
z = Dense(2, activation="softmax")(x)
model = tf.keras.Model(inputs=inputA, outputs=z)
model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=['accuracy'])
model.summary()
train_history = model.fit(
    [x_train], encoded_train_label,validation_data=(x_test, encoded_test_label),epochs=200, batch_size=12,verbose=0)

# Result visualization
accuracy = train_history.history['accuracy']
val_accuracy = train_history.history['val_accuracy']
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.legend(['accuray', 'val_accuracy'])
plt.show()
print('Training accuracy')
print(np.max(accuracy))
print('Classification accuracy')
print(np.max(val_accuracy))
