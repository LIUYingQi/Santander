import cPickle as pickle
import numpy as np
import pandas as pd
import heapq

label_index = ['ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1',
         'ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',
         'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1',
         'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
         'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

label_add = ['ind_cco_fin_ult1 ','ind_cder_fin_ult1 ','ind_cno_fin_ult1 ',
         'ind_ctju_fin_ult1 ','ind_ctma_fin_ult1 ','ind_ctop_fin_ult1 ','ind_ctpp_fin_ult1 ','ind_deco_fin_ult1 ',
         'ind_deme_fin_ult1 ','ind_dela_fin_ult1 ','ind_ecue_fin_ult1 ','ind_fond_fin_ult1 ',
         'ind_plan_fin_ult1 ','ind_pres_fin_ult1 ','ind_reca_fin_ult1 ','ind_tjcr_fin_ult1 ','ind_valo_fin_ult1 ',
         'ind_nomina_ult1 ','ind_nom_pens_ult1 ','ind_recibo_ult1 ']

int_label = [3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,22,23,24]
# int_label = [3,4]


df = pd.read_csv('testset.csv',engine='c')
sub = pd.DataFrame()
sub = pd.DataFrame(df['ncodpers'],dtype=str)
count = sub.count()
print sub
print count

for (x,y) in zip(int_label,label_index):
    file = open(str(x)+'.pkl','rb')
    a = pickle.load(file)
    sub[y] = pd.Series(a[:,1],dtype=np.float32)

print sub
sub['ncodpers']=sub['ncodpers'].astype(np.int64)
sub.sort('ncodpers',inplace=True)

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
        # if row[1] ==True:
        #     str_add += 'ind_cco_fin_ult1 '
        # if row[2] ==True:
        #     str_add += 'ind_cno_fin_ult1 '
        # if row[3] ==True:
        #     str_add += 'ind_ctma_fin_ult1 '
        # if row[4] ==True:
        #     str_add += 'ind_ctop_fin_ult1 '
        # if row[5] ==True:
        #     str_add += 'ind_ctpp_fin_ult1 '
        # if row[6] ==True:
        #     str_add += 'ind_deco_fin_ult1 '
        # if row[7] ==True:
        #     str_add += 'ind_dela_fin_ult1 '
        # if row[8] ==True:
        #     str_add += 'ind_ecue_fin_ult1 '
        # if row[9] ==True:
        #     str_add += 'ind_fond_fin_ult1 '
        # if row[10] ==True:
        #     str_add += 'ind_reca_fin_ult1 '
        if str_sub != '':
            str_sub = str_sub[:len(str_sub)-1]
        str_sub = str(int(row[0]))+','+str_add+'\n'
        print str_sub
        f.write(str_sub)
