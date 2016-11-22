import xgboost as xgb
import pandas as pd
import time
import numpy as np


now = time.time()

dataset = pd.read_csv("2015-01-28_treated.csv")

train = dataset.iloc[:,2:].values

dataset = pd.read_csv("2015-01-28_treated.csv")
labels = dataset.iloc[:,1].values

tests = pd.read_csv("2015-02-28_treated.csv")
#test_id = range(len(tests))
test = tests.iloc[:,2:].values


params={
'booster':'gbtree',

'objective': 'multi:softmax',
'num_class':10,
'gamma':0.05,
'max_depth':12,
'subsample':0.4,
'colsample_bytree':0.7,
'silent':1 ,
'eta': 0.005,
'seed':710,
'nthread':4,
}

plst = list(params.items())

#Using 10000 rows for early stopping.
offset = 35000

num_rounds = 500

xgtest = xgb.DMatrix(test)

xgtrain = xgb.DMatrix(train[:offset,:], label=labels[:offset])
xgval = xgb.DMatrix(train[offset:,:], label=labels[offset:])


watchlist = [(xgtrain, 'train'),(xgval, 'val')]


# training model
model = xgb.train(plst, xgtrain, num_rounds, watchlist,early_stopping_rounds=100)
preds = model.predict(xgtest,ntree_limit=model.best_iteration)

np.savetxt('submission_xgb_MultiSoftmax.csv',np.c_[range(1,len(test)+1),preds],
                delimiter=',',header='ImageId,Label',comments='',fmt='%d')


cost_time = time.time()-now
print "end ......",'\n',"cost time:",cost_time,"(s)......"