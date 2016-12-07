import pandas as pd
import numpy as np
import csv

exist_pro = pd.read_csv('dataset/2016-05-28_treated.csv',engine='c')

label = ['ncodpers','ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

exist_pro = exist_pro.loc[:,label]

exist_pro = pd.DataFrame(exist_pro)
exist_pro.sort_values(by='ncodpers',inplace=True)

f = open('submission.csv','rb')
csv_reader = csv.reader(f)
for i in range(10000):
    csv_reader.next()