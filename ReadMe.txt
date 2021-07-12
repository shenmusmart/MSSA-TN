This is the source code of MSSA-TN proposed in the paper entitled Multi-Source Signal Alignment and 
Efficient Multi-Dimensional Feature Classification in the Application of EEG-Based Subject-Independent Drowsiness Detection.

Requirement:
h5py
Python  version >=3.7.6, 
Tensorflow version >=2.1.0 
Tensornetwork version >=0.2.1

Part of data are packaged to test the algorithm
Original data and extracted muti-dimensional features have been put in data.mat and tensor_feature.mat
Please first run MSSA.py to get the effect of MSSA and multi-dimensional features for tensor network classification
Then run TN.py to see the full result of MSSA-TN

Running classification results:
No adaptation: 0.5
MSSA: 0.57
MSSA-TN: 0.6351 (results may vary for each training)

