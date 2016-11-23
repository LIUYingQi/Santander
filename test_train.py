from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np

mouth_list = ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28','2015-06-28','2015-07-28'
              ,'2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28','2016-01-28','2016-02-28'
              ,'2016-03-28','2016-04-28','2016-05-28']

for i in range(24):

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.7, max_depth=10, random_state=0)

    for item in mouth_list:
        print item

        train_set = pd.read_csv('dataset/'+item+'_treated.csv')
        train_label = pd.read_csv('dataset/'+item+'_label.csv')

        train_set = train_set.iloc[:600000,1:].values
        train_label = train_label.iloc[:600000,i+9].values

        test_set = pd.read_csv('dataset/2016-04-28_treated.csv')
        test_label = pd.read_csv('dataset/2016-04-28_label.csv')

        test_set = test_set.iloc[:5000,1:].values
        test_label = test_label.iloc[:5000,i+9].values

        train_set = np.split(train_set,10,axis=0)
        train_label = np.split(train_label,10,axis=0)

        for batch in range(len(train_set)):
            clf.fit(train_set[batch],train_label[batch])
            print clf.score(test_set, test_label)

    clf.predict(test_set[:5000])