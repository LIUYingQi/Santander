import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier
import cPickle as pickle

print 'loading data set'

data_path = 'input/'


f_train_x = open(data_path+'trainset.pkl','rb')
f_train_y = open(data_path+'label.pkl','rb')

stander = StandardScaler()
trainset = pickle.load(f_train_x)
label = pickle.load(f_train_y)
label = np.array(label,dtype=np.int8)

trainset = stander.fit_transform(trainset)

f_train_x.close()
f_train_y.close()

print trainset.shape
print label.shape

# train_x,test_x,train_y,test_y = cross_validation.train_test_split(trainset,label,test_size=0.1)
#
# print train_x.shape
# print train_y.shape
# print test_x.shape
# print test_y.shape

classifier = OneVsRestClassifier(
    XGBClassifier(n_estimators=27, max_depth=7, min_child_weight=1, colsample_bytree=0.9, subsample=0.8, gamma=0.3,
                  reg_alpha=10, learning_rate=0.2, silent=True))

print 'training ............'
print cross_validation.cross_val_score(classifier,trainset,label,cv = 5)

# print 'grid searching.....................'
#
# for i in [10,20,30,40,100]:
#     print i
#     classifier = OneVsRestClassifier(XGBClassifier(n_estimators=27,max_depth=7,min_child_weight=1,colsample_bytree=0.9,subsample=0.8,gamma=0.3,reg_alpha=10,learning_rate=0.2,silent=True))
#
#     # print 'training ............'
#     # print cross_validation.cross_val_score(classifier,train_x,train_y,cv = 5)
#
#     print 'testing.............'
#     classifier.fit(train_x,train_y)
#     print classifier.score(test_x,test_y)

# classifier = OneVsRestClassifier(XGBClassifier(n_estimators=27,max_depth=7,min_child_weight=1,colsample_bytree=0.9,subsample=0.8,gamma=0.3,reg_alpha=10,learning_rate=0.2,silent=True))
# classifier.fit(trainset,label)
#
# print "Predicting............."
# f_test_x = open(data_path+'test_x.pkl','rb')
# test_X = pickle.load(f_test_x)
#
# test_X = stander.transform(test_X)
#
# preds = classifier.predict_proba(test_X)
# del test_X
#
# result_file = open(data_path+'result.pkl','wb')
# pickle.dump(preds,result_file)
# print preds
# result_file.close()
#
# print 'generating submission file................'
# result_file = open(data_path+'result.pkl','rb')
# preds = pickle.load(result_file)
#
# target_cols = ['ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
#                'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
#                'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
#                'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
#                'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
#
# test_X = pd.read_csv(data_path+'testset.csv',engine='c')
# test_X  =test_X.loc[:,target_cols].values
#
# preds = np.subtract(preds,test_X)
#
# print("Getting the top products..")
# target_cols = np.array(target_cols)
# preds = np.argsort(preds, axis=1)
# preds = np.fliplr(preds)[:, :7]
# print preds
#
# test_id = np.array(pd.read_csv(data_path+"test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
# final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
# out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
# out_df.to_csv('submission.csv', index=False)
#
