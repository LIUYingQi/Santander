import tensorflow as tf
import pandas as pd
import numpy as np
import cPickle as pickle

data_path = "input/"

##################################################################################################################
##############      generate Dmatrix
##################################################################################################################

data_path = "input/"
train_x = pd.read_csv(data_path+'trainset.csv',engine='c').as_matrix()
train_y = pd.read_csv(data_path+'label.csv',engine='c').as_matrix()
train_x = train_x[:,1:]
train_y = train_y[:,3:]

print(train_x.shape, train_y.shape)

test_X = pd.read_csv(data_path+'testset.csv',engine='c').as_matrix()
test_X = test_X[:,1:]
print test_X.shape

trainset = np.empty((0,train_x.shape[1]),dtype=np.float32)
label = np.empty((0),dtype=np.float32)

for i in range(train_x.shape[0]):
    count = -1
    for item in train_y[i]:
        count +=1
        if item == True:
            trainset = np.vstack((trainset,train_x[i]))
            label = np.append(label,count)

print trainset.shape
print label.shape

f_trainset = open(data_path+'trainset.pkl','wb')
f_label = open(data_path+'label.pkl','wb')
f_test_x = open(data_path+'test_x.pkl','wb')
pickle.dump(trainset,f_trainset)
pickle.dump(label,f_label)
pickle.dump(test_X,f_test_x)

#################################################################################################################
#######  model
#################################################################################################################

