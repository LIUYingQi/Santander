import xgboost as xgb
import pandas as pd
import numpy as np
import cPickle as pickle
import sklearn.preprocessing

data_path = "input/"

#
# ##############################################################################################################
# ######               train  model
# #############################################################################################################

f_train_x = open(data_path+'trainset.pkl','rb')
f_train_y = open(data_path+'label.pkl','rb')

stander = sklearn.preprocessing.StandardScaler()
trainset = pickle.load(f_train_x)
label = pickle.load(f_train_y)

trainset = stander.fit_transform(trainset)

f_train_x.close()
f_train_y.close()

def runXGB(trainset, label, seed_val=0):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.2
    param['max_depth'] = 7
    param['silent'] = 0
    param['num_class'] = 22
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.9
    param['seed'] = seed_val
    num_rounds = 27

    plst = list(param.items())
    xgtrain = xgb.DMatrix(trainset, label=label)
    model = xgb.train(plst, xgtrain, num_rounds)
    return model

print("Building model..")
model = runXGB(trainset, label, seed_val=0)

del trainset, label
#
print("Predicting..")

f_test_x = open(data_path+'test_x.pkl','rb')
test_X = pickle.load(f_test_x)
test_X = stander.transform(test_X)
xgtest = xgb.DMatrix(test_X)
preds = model.predict(xgtest)
del test_X, xgtest

result_file = open(data_path+'result.pkl','wb')
pickle.dump(preds,result_file)
print preds

result_file.close()

#####################################################################################################################
####            predicting  and  add  exist product check
####################################################################################################################

result_file = open(data_path+'result.pkl','rb')
preds = pickle.load(result_file)

target_cols = ['ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
               'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

test_X = pd.read_csv(data_path+'testset.csv',engine='c')
test_X  =test_X.loc[:,target_cols].values

preds = np.subtract(preds,test_X)

print("Getting the top products..")
target_cols = np.array(target_cols)
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:, :7]
print preds

test_id = np.array(pd.read_csv(data_path+"test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
out_df.to_csv('submission.csv', index=False)

