import pandas as pd
import csv
import numpy as np
import cPickle as pickle
# fecha_dato,ncodpers,ind_empleado,pais_residencia,sexo,age,fecha_alta,ind_nuevo,antiguedad,indrel
# ,ult_fec_cli_1t,indrel_1mes,tiprel_1mes,indresi,indext,conyuemp,canal_entrada,indfall,tipodom,
# cod_prov,nomprov,ind_actividad_cliente,renta,segmento,

# ind_ahor_fin_ult1,ind_aval_fin_ult1,ind_cco_fin_ult1,ind_cder_fin_ult1,ind_cno_fin_ult1,ind_ctju_fin_ult1,
# ind_ctma_fin_ult1,ind_ctop_fin_ult1,ind_ctpp_fin_ult1,ind_deco_fin_ult1,ind_deme_fin_ult1,ind_dela_fin_ult1,
# ind_ecue_fin_ult1,ind_fond_fin_ult1,ind_hip_fin_ult1,ind_plan_fin_ult1,ind_pres_fin_ult1,ind_reca_fin_ult1,
# ind_tjcr_fin_ult1,ind_valo_fin_ult1,ind_viv_fin_ult1,ind_nomina_ult1,ind_nom_pens_ult1,ind_recibo_ult1

info = pd.read_csv('dataset/2015-05-28_treated.csv')
print info.shape
lable = pd.read_csv('dataset/2015-05-28_label.csv',dtype=np.int32)

labelset = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',
                'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1',
                'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

for item in labelset:
    lable[item+'_add'] = lable[item]
    lable.drop(item,axis=1,inplace=True)

info = pd.concat((info,lable),axis=1)

#  find relation between canal_entrada and product
canal_entrada_unique = info['canal_entrada'].unique()
print canal_entrada_unique

canal_entrada_info = pd.DataFrame()
canal_entrada_info['canal_entrada'] = pd.Series(canal_entrada_unique)
print canal_entrada_info

for product in labelset:
    item_info = []
    for item in canal_entrada_unique:
        item_info.append(info.loc[(info['canal_entrada']==item),product].sum())
    canal_entrada_info['canal_'+product] = pd.Series(item_info)
    print 'continue'
print canal_entrada_info

for product in labelset:
    item_info = []
    for item in canal_entrada_unique:
        item_info.append(info.loc[(info['canal_entrada']==item),product+'_add'].sum())
    canal_entrada_info['canal_'+product] = pd.Series(item_info)
    print 'continue'
print canal_entrada_info

item_info = []

for item in canal_entrada_unique:
    item_info.append(info.loc[(info['canal_entrada']==item),'num_product'].sum())
canal_entrada_info['canal_num_product'] = pd.Series(item_info)
print 'continue'
print canal_entrada_info

f = open('dataset/canal_entrada.pkl','wb')
pickle.dump(canal_entrada_info,f)


# find relation between numprov and product
nomprov_unique = info['nomprov'].unique()

nomprov_info = pd.DataFrame()
nomprov_info['nomprov'] = pd.Series(nomprov_unique)
print nomprov_info

for product in labelset:
    item_info = []
    for item in nomprov_unique:
        item_info.append(info.loc[(info['nomprov']==item),product].sum())
        nomprov_info['nomprov_'+product] = pd.Series(item_info)
    print 'continue'
print nomprov_info

for product in labelset:
    item_info = []
    for item in nomprov_unique:
        item_info.append(info.loc[(info['nomprov']==item),product+'_add'].sum())
        nomprov_info['nomprov_'+product] = pd.Series(item_info)
    print 'continue'
print nomprov_info

item_info = []

for item in nomprov_unique:
    item_info.append(info.loc[(info['nomprov']==item),'num_product'].sum())
nomprov_info['nomprov_num_product'] = pd.Series(item_info)
print 'continue'
print nomprov_info

f = open('dataset/nomprov.pkl','wb')
pickle.dump(nomprov_info,f)