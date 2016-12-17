import pandas as pd
import numpy as np

lables = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1',
           'ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1',
           'ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
           'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
           'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
           'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

# mouth_list = ['2015-02-28','2015-03-28','2015-04-28','2015-05-28','2015-06-28','2015-07-28'
#               ,'2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28','2016-01-28','2016-02-28'
#               ,'2016-03-28','2016-04-28']

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
#
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

general_trainset.to_csv(data_path+'trainset.csv',index=False)
general_label.to_csv(data_path+'label.csv',index=False)