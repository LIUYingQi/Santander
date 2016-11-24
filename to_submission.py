import cPickle as pickle
import numpy as np
import pandas as pd
import csv

label_index = ['ind_cco_fin_ult1' , 'ind_cno_fin_ult1' , 'ind_ctma_fin_ult1' , 'ind_ctop_fin_ult1'
               ,'ind_ctpp_fin_ult1','ind_deco_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1',
               'ind_fond_fin_ult1' ,'ind_reca_fin_ult1']

int_label = [3,5,7,8,9,10,12,13,14,18]

df = pd.read_csv('testset.csv',engine='c')
sub = pd.DataFrame()
sub = pd.DataFrame(df['ncodpers'],dtype=str)
count = sub.count()
print sub
print count

for (x,y) in zip(int_label,label_index):
    file = open(str(x)+'.pkl','rb')
    a = pickle.load(file)
    sub[y] = pd.Series(a,dtype=bool)

print sub
with open('submission.csv','ab') as f:
    f.write('ncodpers,added_products\n')
    sub = sub.iloc[:,:].values
    for row in sub:
        str_add =''
        if row[1]==True:
            str_add = 'ind_cco_fin_ult1'
        if row[2]==True:
            str_add += ' ind_cno_fin_ult1'
        if row[3]==True:
            str_add = 'ind_ctma_fin_ult1'
        if row[4]==True:
            str_add = ' ind_ctop_fin_ult1'
        if row[5]==True:
            str_add = 'ind_ctpp_fin_ult1'
        if row[6]==True:
            str_add = 'ind_deco_fin_ult1'
        if row[7]==True:
            str_add = 'ind_dela_fin_ult1'
        if row[8]==True:
            str_add = 'ind_ecue_fin_ult1'
        if row[9]==True:
            str_add = 'ind_fond_fin_ult1'
        if row[10]==True:
            str_add = 'ind_reca_fin_ult1'
        str = row[0]+','+'\n'
