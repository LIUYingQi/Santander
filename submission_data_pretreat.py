import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('test_ver2.csv',dtype={"sexo":str,"ind_nuevo":str,"ult_fec_cli_1t":str,
                                               "indext":str,"antiguedad":str,"indrel_1mes":str,"tiprel_1mes":str,
                                               "indresi":str,"indext":str,"indfall":str,"conyuemp":str,"renta":str})

# total rows
print len(df['ncodpers'].unique())
count = len(df['ncodpers'].unique())
print df.shape

df_next = pd.read_csv('dataset/2016-05-28.csv',engine='c')
print 'unique df_next   ' + str(len(df_next['ncodpers'].unique()))
print df_next.shape

# ind_ahor_fin_ult1,ind_aval_fin_ult1,ind_cco_fin_ult1,ind_cder_fin_ult1,ind_cno_fin_ult1,ind_ctju_fin_ult1,
# ind_ctma_fin_ult1,ind_ctop_fin_ult1,ind_ctpp_fin_ult1,ind_deco_fin_ult1,ind_deme_fin_ult1,ind_dela_fin_ult1,
# ind_ecue_fin_ult1,ind_fond_fin_ult1,ind_hip_fin_ult1,ind_plan_fin_ult1,ind_pres_fin_ult1,ind_reca_fin_ult1,
# ind_tjcr_fin_ult1,ind_valo_fin_ult1,ind_viv_fin_ult1,ind_nomina_ult1,ind_nom_pens_ult1,ind_recibo_ult1

# add labels
df['ind_ahor_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_aval_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_cco_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_cder_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_cno_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_ctju_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_ctma_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_ctop_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_ctpp_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_deco_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_deme_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_dela_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_ecue_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_fond_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_hip_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_plan_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_pres_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_reca_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_tjcr_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_valo_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_viv_fin_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_nomina_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_nom_pens_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
df['ind_recibo_ult1']=pd.Series(np.zeros(count,dtype=np.int8))
print df.shape

df = df[['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1'
    , 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                   'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1',
                   'ind_fond_fin_ult1'
    , 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1',
                   'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']]

df_next = df_next[['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1'
    , 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                   'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1',
                   'ind_fond_fin_ult1'
    , 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1',
                   'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']]

print df.shape
print df_next.shape

df_change = pd.DataFrame(df.loc[df['ncodpers'].isin(df_next['ncodpers'])])
print 'unique df_next in df  ' + str(len(df_change['ncodpers'].unique()))

df_first_not_in_next = pd.DataFrame(df.loc[~df['ncodpers'].isin(df_next['ncodpers'])])
print 'unique df not in next  ' + str(len(df_first_not_in_next['ncodpers'].unique()))

df_next_not_in_first = pd.DataFrame(df_next.loc[~df_next['ncodpers'].isin(df['ncodpers'])])
print 'unique df next in first  ' + str(len(df_next_not_in_first['ncodpers'].unique()))

df_conct = pd.DataFrame(pd.concat([df,df_next,df_first_not_in_next,df_next_not_in_first]))
print 'unique df  ' + str(len(df['ncodpers'].unique()))

df_grouped = df_conct.groupby('ncodpers', sort=False)
df = df_grouped.last() > df_grouped.first()
df_add = df[0:count]
print df_add.shape

print df_add

# print df_add
da = pd.read_csv('test_ver2.csv',dtype={"sexo":str,"ind_nuevo":str,"ult_fec_cli_1t":str,
                                               "indext":str,"antiguedad":str,"indrel_1mes":str,"tiprel_1mes":str,
                                               "indresi":str,"indext":str,"indfall":str,"conyuemp":str,"renta":str})

print da.shape
for item in ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1'
    , 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                   'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1',
                   'ind_fond_fin_ult1'
    , 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1',
                   'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']:
    da[item]=pd.Series(np.array(df_add[item],dtype=np.int8))
print da

del df
del df_add
del df_next,df_change,df_conct,df_grouped,df_first_not_in_next,df_next_not_in_first



# drop some bad data ( dont worry we got enough data )
da.drop('ult_fec_cli_1t',axis=1,inplace=True)
da.drop(["tipodom","cod_prov"],axis=1,inplace=True)

# chnage time format
da["fecha_dato"] = pd.to_datetime(da["fecha_dato"],format="%Y-%m-%d")
da["fecha_alta"] = pd.to_datetime(da["fecha_alta"],format="%Y-%m-%d")

# sexo
da['sexo_H']=pd.Series(np.zeros(count,dtype=np.int8))
da['sexo_V']=pd.Series(np.zeros(count,dtype=np.int8))
da.loc[da['sexo']=='H','sexo_H'] = 1
da.loc[da['sexo']=='V','sexo_V'] = 1
da.drop('sexo',axis=1,inplace=True)

# ind_empleado
da['ind_empleado_N']=pd.Series(np.zeros(count,dtype=np.int8))
da['ind_empleado_A']=pd.Series(np.zeros(count,dtype=np.int8))
da['ind_empleado_F']=pd.Series(np.zeros(count,dtype=np.int8))
da['ind_empleado_B']=pd.Series(np.zeros(count,dtype=np.int8))
da.loc[da['ind_empleado']=='N','ind_empleado_N'] = 1
da.loc[da['ind_empleado']=='A','ind_empleado_A'] = 1
da.loc[da['ind_empleado']=='F','ind_empleado_F'] = 1
da.loc[da['ind_empleado']=='B','ind_empleado_B'] = 1
da.drop('ind_empleado',axis=1,inplace=True)

# indrel_1mes
da['indrel_1mes_3']=pd.Series(np.zeros(count,dtype=np.int8))
da.loc[da['indrel_1mes']=='3','indrel_1mes_3'] = 1
da.drop('indrel_1mes',axis=1,inplace=True)

# tiprel_1mes
da['tiprel_1mes_A']=pd.Series(np.zeros(count,dtype=np.int8))
da['tiprel_1mes_I']=pd.Series(np.zeros(count,dtype=np.int8))
da.loc[da['tiprel_1mes']=='A','tiprel_1mes_A'] = 1
da.loc[da['tiprel_1mes']=='I','tiprel_1mes_I'] = 1
da.drop('tiprel_1mes',axis=1,inplace=True)

# segmento
da['segmento_1']=pd.Series(np.zeros(count,dtype=np.int8))
da['segmento_2']=pd.Series(np.zeros(count,dtype=np.int8))
da['segmento_3']=pd.Series(np.zeros(count,dtype=np.int8))
da.loc[da['segmento']=='01 - TOP','segmento_1'] = 1
da.loc[da['segmento']=='02 - PARTICULARES','segmento_2'] = 1
da.loc[da['segmento']=='03 - UNIVERSITARIO','segmento_3'] = 1
da.drop('segmento',axis=1,inplace=True)

# indresi
da['indresi_N']=pd.Series(np.zeros(count,dtype=np.int8))
da.loc[da['indresi']=='N','indresi_N'] = 1
da.drop('indresi',axis=1,inplace=True)

# indext
da['indext_S']=pd.Series(np.zeros(count,dtype=np.int8))
da.loc[da['indext']=='S','indext_S'] = 1
da.drop('indext',axis=1,inplace=True)

# indext
da['indfall_S']=pd.Series(np.zeros(count,dtype=np.int8))
da.loc[da['indfall']=='S','indfall_S'] = 1
da.drop('indfall',axis=1,inplace=True)

# conyuemp
da['conyuemp_S']=pd.Series(np.zeros(count,dtype=np.int8))
da.loc[da['conyuemp']=='S','conyuemp'] = 1
da.drop('conyuemp',axis=1,inplace=True)

# age to int
da["age"] = pd.to_numeric(da["age"], errors="coerce")
da.loc[da.age < 18,"age"] = da.loc[(da.age >= 18) & (da.age <= 30),"age"].mean(skipna=True)
da.loc[da.age > 100,"age"] = da.loc[(da.age >= 30) & (da.age <= 100),"age"].mean(skipna=True)
da["age"].fillna(da["age"].mean(),inplace=True)
da["age"] = da["age"].astype(int)

# ind_neuvo
da.loc[da["ind_nuevo"].isnull(),"ind_nuevo"] = 0

# antiguedad
da.loc[da.antiguedad=='     NA',"antiguedad"] = '0'
da.loc[da.antiguedad=='-999999', "antiguedad"] = '0'
da["antiguedad"] = da["antiguedad"].astype(int)

print da
print '*********************************************************************'
# fecha_alta

dates=da.loc[:,"fecha_alta"].sort_values().reset_index()
median_date = int(np.median(dates.index.values))
da.loc[da.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]

print da

# indrel
print da['indrel'].unique()
da.loc[da.indrel.isnull(),"indrel"] = 1
da['indrel_change'] = pd.Series(np.zeros(count, dtype=np.int8))
da.loc[da['indrel'] == 99, 'indrel_change'] = 1
da.drop('indrel', axis=1, inplace=True)

# ind_actividad_cliente
da.loc[da.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = da["ind_actividad_cliente"].min()
da['ind_actividad_cliente'] = da['ind_actividad_cliente'].astype(np.int8)

# nomprovince
da.loc[da.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
da.loc[da.nomprov.isnull(),"nomprov"] = "UNKNOWN"

# income
da.loc[da.renta=='         NA',"renta"] = 128544.0
da["renta"] = da["renta"].astype(float)

incomes = da.loc[da.renta.notnull(),:].groupby("nomprov").agg({"renta":{"MedianIncome":np.median}})
incomes.sort_values(by=("renta","MedianIncome"),inplace=True)
incomes.reset_index(inplace=True)
incomes.nomprov = incomes.nomprov.astype("category", categories=[i for i in da.nomprov.unique()],ordered=False)

grouped = da.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
new_incomes = pd.merge(da,grouped,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]
new_incomes = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")
da.sort_values("nomprov",inplace=True)
da = da.reset_index()
new_incomes = new_incomes.reset_index()

da.loc[da.renta.isnull(),"renta"] = new_incomes.loc[da.renta.isnull(),"renta"].reset_index()
da.loc[da.renta.isnull(),"renta"] = da.loc[da.renta.notnull(),"renta"].median()
print '#############################################'
print da
da.sort_values(by="fecha_dato",inplace=True)
print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
print da
string_data = da.select_dtypes(include=["object"])
missing_columns = [col for col in string_data if string_data[col].isnull().any()]
for col in missing_columns:
    print("Unique values for {0}:\n{1}\n".format(col,string_data[col].unique()))
del string_data
da.sort_values(by="index",inplace=True)

del da['index']

da["year"] = pd.DatetimeIndex(da["fecha_dato"]).year
da["mouth"] = pd.DatetimeIndex(da["fecha_dato"]).month
da["year_fecha"] = pd.DatetimeIndex(da["fecha_alta"]).year
da["mouth_fecha"] = pd.DatetimeIndex(da["fecha_alta"]).month
da['fecha_diff'] = (da["year"] - da["year_fecha"]) * 12 + da["mouth"] - da["mouth_fecha"]
da.drop(['fecha_dato','fecha_alta','year','year_fecha'],axis=1,inplace=True)

canal_entrada_label = ['KHE', 'KAT', 'KFC', 'KFA', 'KHK', 'KHD', 'KAS', 'KAG', 'RED',
                       'KAA', 'KAY', 'KAB', 'KHN', 'KHL', 'KCC', 'KAE', 'KBZ', 'KFD',
                       'KHM', 'KAI', 'KEY', 'KAW', 'KAF', 'KAR', '013', 'KAZ', 'KAH',
                       'KCI', 'KCH', 'KAJ']
for item in canal_entrada_label:
    da[item] = pd.Series(np.zeros(count, dtype=np.int8))
    da.loc[da['canal_entrada'] == item, item] = 1

nomprov_label = ['ALAVA','MADRID','MALAGA','MURCIA','NAVARRA','OURENSE','MELILLA','VALENCIA','UNKNOWN','TOLEDO',
                 'TERUEL','ZARAGOZA','ZAMORA','VALLADOLID','PONTEVEDRA','RIOJA, LA','SALAMANCA','PALMAS, LAS',
                 'PALENCIA','SEVILLA','TARRAGONA','SORIA','SEGOVIA','SANTA CRUZ DE TENERIFE','BARCELONA',
                 'BIZKAIA','BURGOS','CANTABRIA','CASTELLON','CADIZ','CIUDAD REAL','CORDOBA','CEUTA','CACERES',
                 'ASTURIAS','ALMERIA','ALICANTE','AVILA','BADAJOZ','ALBACETE','BALEARS, ILLES','CORUNA, A','CUENCA','GIRONA',
                 'GRANADA','GIPUZKOA','LUGO','LERIDA','LEON','GUADALAJARA','HUELVA','JAEN','HUESCA']
for item in nomprov_label:
    da[item] = pd.Series(np.zeros(count, dtype=np.int8))
    da.loc[da['nomprov'] == item, item] = 1

da.drop(['pais_residencia','canal_entrada','nomprov'],axis=1,inplace=True)

print da.shape

da.to_csv('testset.csv',index=False)