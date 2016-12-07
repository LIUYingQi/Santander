from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import cPickle as pickle

# mouth_list = ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28','2015-06-28','2015-07-28'
#               ,'2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28','2016-01-28','2016-02-28'
#               ,'2016-03-28','2016-04-28']

mouth_list = ['2015-05-28','2016-03-28','2016-04-28']
mouth_len_list=[]
for mouth in mouth_list:
    train_set = pd.read_csv('dataset/' + mouth + '_treated.csv', engine='c')
    mouth_len_list.append(train_set.shape[0])
    print 'continue'

print mouth_len_list

label_index = ['ind_plan_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
# ind_tjcr_fin_ult1 ind_valo_fin_ult1 ind_nomina_ult1  ind_nom_pens_ult1  ind_recibo_ult1]

title = ['ncodpers','ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1',
         'ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',
         'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
         'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
         'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

#ind_ahor_fin_ult1,ind_aval_fin_ult1,ind_cco_fin_ult1,ind_cder_fin_ult1,ind_cno_fin_ult1,ind_ctju_fin_ult1,ind_ctma_fin_ult1,
# ind_ctop_fin_ult1,ind_ctpp_fin_ult1,ind_deco_fin_ult1,ind_deme_fin_ult1,ind_dela_fin_ult1,ind_ecue_fin_ult1,ind_fond_fin_ult1,
# ind_hip_fin_ult1,ind_plan_fin_ult1,ind_pres_fin_ult1,ind_reca_fin_ult1,ind_tjcr_fin_ult1,ind_valo_fin_ult1,ind_viv_fin_ult1,
# ind_nomina_ult1,ind_nom_pens_ult1,ind_recibo_ult1

batch_size = 300000

index = []
for item in label_index:
    index.append(title.index(item))

print index

for item_label in index:

    clf = GradientBoostingClassifier(n_estimators=50, warm_start=True)
    # clf = RandomForestClassifier(n_estimators=25,min_samples_split=10,warm_start=True)
    # clf = RandomForestClassifier(n_estimators=100)

    for item in zip(mouth_list,mouth_len_list):
        print item[0]
        print item[1]
        print item_label

        for i_batch in range(item[1]/batch_size):
        # for i_batch in range(1):
            print i_batch

            data_set = pd.read_csv('dataset/'+item[0]+'_treated.csv',engine='c')
            data_label = pd.read_csv('dataset/'+item[0]+'_label.csv',engine='c')

            # train_set = data_set.iloc[:,1:].values
            # train_label = data_label.iloc[:,item_label].values

            train_set = data_set.iloc[i_batch*batch_size:(i_batch+1)*batch_size,1:].values
            train_label = data_label.iloc[i_batch*batch_size:(i_batch+1)*batch_size,item_label].values

            test_set = data_set.iloc[item[1]-10000:item[1],1:].values
            test_label = data_label.iloc[item[1]-10000:item[1],item_label].values

            train_set = np.array(train_set).copy(order='C')
            train_label = np.array(train_label).copy(order='C')
            test_set = np.array(test_set).copy(order='C')
            test_label = np.array(test_label).copy(order='C')

            del data_set
            del data_label

            print train_set.shape
            print train_label.shape

            clf.fit(train_set,train_label)
            n_est = clf.n_estimators
            clf.set_params(n_estimators=n_est+50)
            print clf.score(test_set, test_label)

    sub_set = pd.read_csv('testset.csv', engine='c')
    sub_set = sub_set.iloc[:, 1:].values
    f = file(str(item_label)+'.pkl','wb')
    pickle.dump(clf.predict_proba(sub_set),f)
    del sub_set
    f.close()