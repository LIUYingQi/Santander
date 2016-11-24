from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import cPickle as pickle

# mouth_list = ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28','2015-06-28','2015-07-28'
#               ,'2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28','2016-01-28','2016-02-28'
#               ,'2016-03-28','2016-04-28','2016-05-28']

mouth_list = ['2015-06-28']

label_index = ['ind_cco_fin_ult1' , 'ind_cno_fin_ult1' , 'ind_ctma_fin_ult1' , 'ind_ctop_fin_ult1'
               ,'ind_ctpp_fin_ult1','ind_deco_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1',
               'ind_fond_fin_ult1' ,'ind_reca_fin_ult1']
# ind_tjcr_fin_ult1 ind_valo_fin_ult1 ind_nomina_ult1  ind_nom_pens_ult1  ind_recibo_ult1]

title = ['ncodpers','ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1',
         'ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',
         'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
         'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
         'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

sub_set = pd.read_csv('testset.csv',engine='c')
sub_set = sub_set.iloc[:,1:].values

index = []
for item in label_index:
    index.append(title.index(item))

print index

for item_label in index:

    # clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5, max_depth=5, random_state=0)
    clf = RandomForestClassifier(n_estimators=200,warm_start=True)

    for item in mouth_list:
        print item
        print item_label
        train_set = pd.read_csv('dataset/'+item+'_treated.csv',engine='c')
        train_label = pd.read_csv('dataset/'+item+'_label.csv',engine='c')

        train_set = train_set.iloc[:600000,1:].values
        train_label = train_label.iloc[:600000,item_label].values

        test_set = pd.read_csv('dataset/2016-04-28_treated.csv', engine='c')
        test_label = pd.read_csv('dataset/2016-04-28_label.csv', engine='c')

        test_set = test_set.iloc[:50000, 1:].values
        test_label = test_label.iloc[:50000, item_label].values

        # train_set = np.split(train_set,5,axis=0)
        # train_label = np.split(train_label,5,axis=0)

        # for batch in range(len(train_set)):
        clf.fit(train_set,train_label)
        n_est = clf.n_estimators
        # clf.fit(train_set[batch],train_label[batch])
        clf.set_params(n_estimators=n_est+200)
        print clf.score(test_set, test_label)

    f = file(str(item_label)+'.pkl','wb')
    pickle.dump(clf.predict(sub_set),f)