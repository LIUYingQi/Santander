from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import cPickle as pickle

# label_index = ['ind_deco_fin_ult1',
#          'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
#          'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
#          'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

label_index = ['ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1',
         'ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',
         'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
         'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
         'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

title = ['ncodpers','ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1',
         'ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',
         'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
         'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
         'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

index = []
for item in label_index:
    index.append(title.index(item))

print index

batch = 220000
len = 446510
# len = 23782

for item_label in index:
    print item_label
    # clf = GradientBoostingClassifier(n_estimators=100,min_samples_split=50,warm_start=True)
    clf = RandomForestClassifier(n_estimators=100,min_samples_split=10,warm_start=True)
    # clf = RandomForestClassifier(n_estimators=100)

    for i in range(len/batch):
        print i

        data_set = pd.read_csv('trainset.csv', engine='c')
        data_label = pd.read_csv('label.csv', engine='c')

        data_set.sort_values(by='ncodpers', inplace=True)
        data_set.sort_values(by='ncodpers', inplace=True)

        train_set = data_set.iloc[i*batch:(i+1)*batch,1:].values
        train_label = data_label.iloc[i*batch:(i+1)*batch,item_label].values

        # train_set = data_set.iloc[:,1:].values
        # train_label = data_label.iloc[:,item_label].values

        test_set = data_set.iloc[len-10000:len,1:].values
        test_label = data_label.iloc[len-10000:len,item_label].values

        del data_set
        del data_label

        print train_set.shape
        print train_label.shape

        train_set = np.array(train_set).copy(order='C')
        train_label = np.array(train_label).copy(order='C')

        test_set = np.array(test_set).copy(order='C')
        test_label = np.array(test_label).copy(order='C')

        clf.fit(train_set,train_label)
        n_est = clf.n_estimators
        clf.set_params(n_estimators=n_est+100)
        print clf.score(test_set, test_label)

        sub_set = pd.read_csv('testset.csv', engine='c')
        sub_set = sub_set.iloc[:, 1:].values
        f = file(str(item_label)+'.pkl','wb')
        pickle.dump(clf.predict_proba(sub_set),f)
        del sub_set
        f.close()