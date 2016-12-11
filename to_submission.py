import cPickle as pickle
import numpy as np
import pandas as pd
import heapq

#ncodpers,ind_ahor_fin_ult1,ind_aval_fin_ult1,ind_cco_fin_ult1,ind_cder_fin_ult1,
# ind_cno_fin_ult1,ind_ctju_fin_ult1,ind_ctma_fin_ult1,ind_ctop_fin_ult1,ind_ctpp_fin_ult1,
# ind_deco_fin_ult1,ind_deme_fin_ult1,ind_dela_fin_ult1,ind_ecue_fin_ult1,ind_fond_fin_ult1,
# ind_hip_fin_ult1,ind_plan_fin_ult1,ind_pres_fin_ult1,ind_reca_fin_ult1,ind_tjcr_fin_ult1,
# ind_valo_fin_ult1,ind_viv_fin_ult1,ind_nomina_ult1,ind_nom_pens_ult1,ind_recibo_ult1

#######################################################################################################################
##### hey  fuckkkkkkkkkkkkkk   change   allllllllllllll    hereeeeeeeeeeeeeeeeeeee

exist_pro = pd.read_csv('dataset/2016-05-28_treated.csv',engine='c')

label = ['ncodpers','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1',
         'ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',
         'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
         'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
         'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

label_index = ['ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1',
         'ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',
         'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
         'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
         'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

label_add = ['ind_cco_fin_ult1 ','ind_cder_fin_ult1 ','ind_cno_fin_ult1 ',
         'ind_ctju_fin_ult1 ','ind_ctma_fin_ult1 ','ind_ctop_fin_ult1 ','ind_ctpp_fin_ult1 ','ind_deco_fin_ult1 ',
         'ind_deme_fin_ult1 ','ind_dela_fin_ult1 ','ind_ecue_fin_ult1 ','ind_fond_fin_ult1 ','ind_hip_fin_ult1 ',
         'ind_plan_fin_ult1 ','ind_pres_fin_ult1 ','ind_reca_fin_ult1 ','ind_tjcr_fin_ult1 ','ind_valo_fin_ult1 ',
         'ind_viv_fin_ult1 ','ind_nomina_ult1 ','ind_nom_pens_ult1 ','ind_recibo_ult1 ']

int_label = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
# int_label = [3,4]
######################################################################################################################

exist_pro = exist_pro.loc[:,label]

exist_pro = pd.DataFrame(exist_pro)
exist_pro.sort_values(by='ncodpers',inplace=True)
# print exist_pro.describe
print exist_pro.shape

df = pd.read_csv('testset.csv',engine='c')
sub = pd.DataFrame()
sub = pd.DataFrame(df['ncodpers'],dtype=str)
count = sub.count()
# print sub
# print count

for (x,y) in zip(int_label,label_index):
    file = open(str(x)+'.pkl','rb')
    a = pickle.load(file)
    sub[y] = pd.Series(a[:,1],dtype=np.float32)

sub['ncodpers']=sub['ncodpers'].astype(np.int64)
sub.sort('ncodpers',inplace=True)
# print sub.describe
print sub.shape
del df

df_first_not_in_next = pd.DataFrame(sub.loc[~sub['ncodpers'].isin(exist_pro['ncodpers'])])
print 'unique df not in next  ' + str(len(df_first_not_in_next['ncodpers'].unique()))

df_next_not_in_first = pd.DataFrame(exist_pro.loc[~exist_pro['ncodpers'].isin(sub['ncodpers'])])
print 'unique df next in first  ' + str(len(df_next_not_in_first['ncodpers'].unique()))

df_conct = pd.DataFrame(pd.concat([sub, exist_pro, df_first_not_in_next, df_next_not_in_first]))
print 'unique df  ' + str(len(sub['ncodpers'].unique()))

df_grouped = df_conct.groupby('ncodpers', sort=False)
df = df_grouped.first() - df_grouped.last()
df = pd.DataFrame(df[0:sub.shape[0]])
print df.shape
print df.describe

with open('submission.csv','wb') as f:
    f.write('ncodpers,added_products\n')
    sub = sub.iloc[:,:].values
    for row in sub:
        str_sub=''
        str_add=''
        proba = np.array(row[1:],dtype=float)
        index = heapq.nlargest(7,range(len(proba)),proba.take)
        for item_add in index:
            str_add += label_add[item_add]
        if str_sub != '':
            str_sub = str_sub[:len(str_sub)-1]
        str_sub = str(int(row[0]))+','+str_add+'\n'
        print str_sub
        f.write(str_sub)
