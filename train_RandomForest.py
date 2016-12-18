import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier

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

##################################################################################################################
##############      random  forest
##################################################################################################################

