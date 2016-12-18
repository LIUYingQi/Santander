import pandas as pd
import numpy as np
import cPickle as pickle

print 'loading from dataset and seleting useful label instance '

lables = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1',
           'ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1',
           'ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
           'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
           'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
           'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

# mouth_list = ['2015-02-28','2015-03-28','2015-04-28','2015-05-28','2015-06-28','2015-07-28'
#               ,'2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28','2016-01-28','2016-02-28'
#               ,'2016-03-28','2016-04-28']

mouth_list = ['2015-05-28']

mouth = '2015-05-28'

data_path = "input/"
trainset = pd.read_csv('dataset/'+mouth+'_treated.csv',engine='c')
label = pd.read_csv('dataset/'+mouth+'_label.csv',engine='c')
print label.shape

rm_list = label['ncodpers'][(label.ind_ahor_fin_ult1 == False) &
                            (label.ind_aval_fin_ult1 == False) &
                            (label.ind_cco_fin_ult1 == False) &
                            (label.ind_cder_fin_ult1 == False) &
                            (label.ind_cno_fin_ult1 == False) &
                            (label.ind_ctju_fin_ult1 == False) &
                            (label.ind_ctma_fin_ult1 ==False) &
                            (label.ind_ctop_fin_ult1 == False) &
                            (label.ind_ctpp_fin_ult1 == False) &
                            (label.ind_deco_fin_ult1 == False) &
                            (label.ind_deme_fin_ult1 == False) &
                            (label.ind_dela_fin_ult1 == False) &
                            (label.ind_ecue_fin_ult1 == False) &
                            (label.ind_fond_fin_ult1 == False) &
                            (label.ind_hip_fin_ult1 == False) &
                            (label.ind_plan_fin_ult1 == False) &
                            (label.ind_pres_fin_ult1 == False) &
                            (label.ind_reca_fin_ult1 == False) &
                            (label.ind_tjcr_fin_ult1 == False) &
                            (label.ind_valo_fin_ult1 == False) &
                            (label.ind_viv_fin_ult1 == False) &
                            (label.ind_nomina_ult1 == False) &
                            (label.ind_nom_pens_ult1 == False) &
                            (label.ind_recibo_ult1 == False)]
rm_list = rm_list.values
print rm_list.shape

general_label = label[~label['ncodpers'].isin(rm_list)]
general_trainset = trainset[~trainset['ncodpers'].isin(rm_list)]

print trainset.shape
print label.shape

###########        if add  other  mouth
# for mouth in mouth_list:
#
#     trainset = pd.read_csv('dataset/'+mouth+'_treated.csv',engine='c')
#     label = pd.read_csv('dataset/'+mouth+'_label.csv',engine='c')
#     print label.shape
#
#     rm_list = label['ncodpers'][(label.ind_ahor_fin_ult1 == False) &
#                             (label.ind_aval_fin_ult1 == False) &
#                             (label.ind_cco_fin_ult1 == False) &
#                             (label.ind_cder_fin_ult1 == False) &
#                             (label.ind_cno_fin_ult1 == False) &
#                             (label.ind_ctju_fin_ult1 == False) &
#                             (label.ind_ctma_fin_ult1 ==False) &
#                             (label.ind_ctop_fin_ult1 == False) &
#                             (label.ind_ctpp_fin_ult1 == False) &
#                             (label.ind_deco_fin_ult1 == False) &
#                             (label.ind_deme_fin_ult1 == False) &
#                             (label.ind_dela_fin_ult1 == False) &
#                             (label.ind_ecue_fin_ult1 == False) &
#                             (label.ind_fond_fin_ult1 == False) &
#                             (label.ind_hip_fin_ult1 == False) &
#                             (label.ind_plan_fin_ult1 == False) &
#                             (label.ind_pres_fin_ult1 == False) &
#                             (label.ind_reca_fin_ult1 == False) &
#                             (label.ind_tjcr_fin_ult1 == False) &
#                             (label.ind_valo_fin_ult1 == False) &
#                             (label.ind_viv_fin_ult1 == False) &
#                             (label.ind_nomina_ult1 == False) &
#                             (label.ind_nom_pens_ult1 == False) &
#                             (label.ind_recibo_ult1 == False)]
#     rm_list = rm_list.values
#     print rm_list.shape
#
#     label = label[~label['ncodpers'].isin(rm_list)]
#     trainset = trainset[~trainset['ncodpers'].isin(rm_list)]
#
#     print trainset.shape
#     print label.shape
#
#     general_trainset = pd.DataFrame(pd.concat([general_trainset,trainset]))
#     general_label = pd.DataFrame(pd.concat([general_label,label]))

print general_label.shape
print general_trainset.shape

######     handle multiclass label problem
print 'multiclass label to one class label'

data_path = "input/"
train_x = general_trainset.values
train_y = general_label.values
train_x = train_x[:,1:]
train_y = train_y[:,3:]

print(train_x.shape, train_y.shape)

test_X = pd.read_csv(data_path+'testset.csv',engine='c').as_matrix()
test_X = test_X[:,1:]
print test_X.shape


#####   multi label
f_trainset = open(data_path+'trainset.pkl','wb')
f_label = open(data_path+'label.pkl','wb')
f_test_x = open(data_path+'test_x.pkl','wb')
pickle.dump(train_x,f_trainset)
pickle.dump(train_y,f_label)
pickle.dump(test_X,f_test_x)


######   xgboost
# trainset = np.empty((0,train_x.shape[1]),dtype=np.float32)
# label = np.empty((0),dtype=np.float32)
#
# for i in range(train_x.shape[0]):
#     count = -1
#     for item in train_y[i]:
#         count +=1
#         if item == True:
#             trainset = np.vstack((trainset,train_x[i]))
#             label = np.append(label,count)
#
# print trainset.shape
# print label.shape

# f_trainset = open(data_path+'trainset.pkl','wb')
# f_label = open(data_path+'label.pkl','wb')
# f_test_x = open(data_path+'test_x.pkl','wb')
# pickle.dump(trainset,f_trainset)
# pickle.dump(label,f_label)
# pickle.dump(test_X,f_test_x)