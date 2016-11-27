import pandas as pd

# fecha_dato,ncodpers,ind_empleado,pais_residencia,sexo,age,fecha_alta,ind_nuevo,antiguedad,indrel
# ,ult_fec_cli_1t,indrel_1mes,tiprel_1mes,indresi,indext,conyuemp,canal_entrada,indfall,tipodom,
# cod_prov,nomprov,ind_actividad_cliente,renta,segmento,
# ind_ahor_fin_ult1,ind_aval_fin_ult1,ind_cco_fin_ult1,ind_cder_fin_ult1,ind_cno_fin_ult1,ind_ctju_fin_ult1,
# ind_ctma_fin_ult1,ind_ctop_fin_ult1,ind_ctpp_fin_ult1,ind_deco_fin_ult1,ind_deme_fin_ult1,ind_dela_fin_ult1,
# ind_ecue_fin_ult1,ind_fond_fin_ult1,ind_hip_fin_ult1,ind_plan_fin_ult1,ind_pres_fin_ult1,ind_reca_fin_ult1,
# ind_tjcr_fin_ult1,ind_valo_fin_ult1,ind_viv_fin_ult1,ind_nomina_ult1,ind_nom_pens_ult1,ind_recibo_ult1

train_file = 'dataset/2015-06-28_label.csv'
info = pd.read_csv(train_file,engine='c',dtype=float)
# print info.describe()
info1 = info[['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1']]
info2 = info[['ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1']]
info3 = info[['ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1']]
info4 = info[['ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']]
# # ind_ctma_fin_ult1,ind_ctop_fin_ult1,ind_ctpp_fin_ult1]
# # print info['ind_ahor_fin_ult1'].describe
# # print info.head()

print info1.describe()
print info2.describe()
print info3.describe()
print info4.describe()



# print info['tiprel_1mes'].unique()
# print info['indresi_N'].sum()
# print info['indresi_S'].sum()
# print info['indext_N'].sum()
# print info['indext_S'].sum()
#
# print info['indresi'].unique()
# print info['indext'].unique()
# # # print info['sexo'].unique()
# # print info['sexo_H'].sum()
# print info['sexo_U'].sum()
# print info['sexo_V'].sum()
#
#
# # print info.groupby(['sexo']).count()
# # print 'ind_empleado'
# print info['ind_empleado'].unique()
# print 'pais_residencia'
# print info['pais_residencia'].unique()
# print 'sexo'
# print info['sexo'].unique()
# print 'antiguedad'
# print info['antiguedad'].unique()
# print 'ult_fec_cli_1t'
# print info['ult_fec_cli_1t'].unique()
# print 'indrel_1mes'
# print info['indrel_1mes'].unique()
# print 'tiprel_1mes'
# print info['tiprel_1mes'].unique()
# print 'indresi'
# print info['indresi'].unique()
# print 'indext'
# print info['indext'].unique()
# print 'conyuemp'
# print info['conyuemp'].unique()
# print 'canal_entrada'
# print info['canal_entrada'].unique()
# print 'indfall'
# print info['indfall'].unique()
# print 'nomprov'
# print info['nomprov'].unique()
# print 'segmento'
# print info['segmento'].unique()
#
# print info['ind_actividad_cliente']

# print info['ind_empleado'].unique()
# print info.dtypes
#
# print info.isnull().any()

# info['sexo'] = info['sexo'].astype('category')
# print info['sexo'].cat.categories
#
# info['ind_viv_fin_ult1'] = info['ind_viv_fin_ult1'].astype('category')
# print info['ind_viv_fin_ult1'].cat.categories
#
# info['renta'] = info['renta'].astype('category')
# print info['renta'].cat.categories

#
# train_file = 'test_ver2.csv'
# info = pd.read_csv(train_file,engine='c')
# print info['renta']
# print info['renta'].unique()