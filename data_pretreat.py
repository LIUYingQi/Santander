import numpy as np
import pandas as pd


mouth_list = ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28','2015-06-28','2015-07-28'
              ,'2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28','2016-01-28','2016-02-28'
              ,'2016-03-28','2016-04-28','2016-05-28']

# mouth_list = ['2015-06-28']

# ind_ahor_fin_ult1,ind_aval_fin_ult1,ind_cco_fin_ult1,ind_cder_fin_ult1,ind_cno_fin_ult1,ind_ctju_fin_ult1,
# ind_ctma_fin_ult1,ind_ctop_fin_ult1,ind_ctpp_fin_ult1,ind_deco_fin_ult1,ind_deme_fin_ult1,ind_dela_fin_ult1,
# ind_ecue_fin_ult1,ind_fond_fin_ult1,ind_hip_fin_ult1,ind_plan_fin_ult1,ind_pres_fin_ult1,ind_reca_fin_ult1,
# ind_tjcr_fin_ult1,ind_valo_fin_ult1,ind_viv_fin_ult1,ind_nomina_ult1,ind_nom_pens_ult1,ind_recibo_ult1

for mouth_item in mouth_list:
    print mouth_item
    df = pd.read_csv('dataset/'+mouth_item+".csv",dtype={"sexo":str,"ind_nuevo":str,"ult_fec_cli_1t":str,
                                         "indext":str,"antiguedad":str,"indrel_1mes":str,"tiprel_1mes":str,
                                                   "indresi":str,"indext":str,"indfall":str,"conyuemp":str})

    # total rows
    print len(df['ncodpers'].unique())
    count = len(df['ncodpers'].unique())

    # drop some bad data ( dont worry we got enough data )
    df.drop('ult_fec_cli_1t',axis=1,inplace=True)
    df.drop(["tipodom","cod_prov"],axis=1,inplace=True)

    # chnage time format
    df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
    df["fecha_alta"] = pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")

    # sexo
    df['sexo_H']=pd.Series(np.zeros(count,dtype=np.int8))
    df['sexo_V']=pd.Series(np.zeros(count,dtype=np.int8))
    df.loc[df['sexo']=='H','sexo_H'] = 1
    df.loc[df['sexo']=='V','sexo_V'] = 1
    df.drop('sexo',axis=1,inplace=True)

    # ind_empleado
    df['ind_empleado_N']=pd.Series(np.zeros(count,dtype=np.int8))
    df['ind_empleado_A']=pd.Series(np.zeros(count,dtype=np.int8))
    df['ind_empleado_F']=pd.Series(np.zeros(count,dtype=np.int8))
    df['ind_empleado_B']=pd.Series(np.zeros(count,dtype=np.int8))
    df.loc[df['ind_empleado']=='N','ind_empleado_N'] = 1
    df.loc[df['ind_empleado']=='A','ind_empleado_A'] = 1
    df.loc[df['ind_empleado']=='F','ind_empleado_F'] = 1
    df.loc[df['ind_empleado']=='B','ind_empleado_B'] = 1
    df.drop('ind_empleado',axis=1,inplace=True)

    # indrel_1mes

    df['indrel_1mes_3']=pd.Series(np.zeros(count,dtype=np.int8))
    df.loc[df['indrel_1mes']=='3','indrel_1mes_3'] = 1
    df.drop('indrel_1mes',axis=1,inplace=True)

    # tiprel_1mes
    df['tiprel_1mes_A']=pd.Series(np.zeros(count,dtype=np.int8))
    df['tiprel_1mes_I']=pd.Series(np.zeros(count,dtype=np.int8))
    df.loc[df['tiprel_1mes']=='A','tiprel_1mes_A'] = 1
    df.loc[df['tiprel_1mes']=='I','tiprel_1mes_I'] = 1
    df.drop('tiprel_1mes',axis=1,inplace=True)

    # segmento
    df['segmento_1']=pd.Series(np.zeros(count,dtype=np.int8))
    df['segmento_2']=pd.Series(np.zeros(count,dtype=np.int8))
    df['segmento_3']=pd.Series(np.zeros(count,dtype=np.int8))
    df.loc[df['segmento']=='01 - TOP','segmento_1'] = 1
    df.loc[df['segmento']=='02 - PARTICULARES','segmento_2'] = 1
    df.loc[df['segmento']=='03 - UNIVERSITARIO','segmento_3'] = 1
    df.drop('segmento',axis=1,inplace=True)

    # indresi
    df['indresi_N']=pd.Series(np.zeros(count,dtype=np.int8))
    df.loc[df['indresi']=='N','indresi_N'] = 1
    df.drop('indresi',axis=1,inplace=True)

    # indext
    df['indext_S']=pd.Series(np.zeros(count,dtype=np.int8))
    df.loc[df['indext']=='S','indext_S'] = 1
    df.drop('indext',axis=1,inplace=True)

    # indext
    df['indfall_S']=pd.Series(np.zeros(count,dtype=np.int8))
    df.loc[df['indfall']=='S','indfall_S'] = 1
    df.drop('indfall',axis=1,inplace=True)

    # conyuemp
    df['conyuemp_S']=pd.Series(np.zeros(count,dtype=np.int8))
    df.loc[df['conyuemp']=='S','conyuemp'] = 1
    df.drop('conyuemp',axis=1,inplace=True)

    # age to int
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[df.age < 18,"age"] = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
    df.loc[df.age > 100,"age"] = df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True)
    df["age"].fillna(df["age"].mean(),inplace=True)
    df["age"] = df["age"].astype(np.int8)

    # ind_neuvo
    df.loc[df["ind_nuevo"].isnull(),"ind_nuevo"] = 0
    df["ind_nuevo"] = df["ind_nuevo"].astype(np.int8)

    # antiguedad
    df.loc[df.antiguedad=='     NA',"antiguedad"] = '0'
    df.loc[df.antiguedad=='-999999', "antiguedad"] = '0'
    df["antiguedad"] = df["antiguedad"].astype(np.int64)

    # fecha_alta
    dates=df.loc[:,"fecha_alta"].sort_values().reset_index()
    median_date = int(np.median(dates.index.values))
    df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]

    # indrel
    print df['indrel'].unique()
    df.loc[df.indrel.isnull(),"indrel"] = 1
    df['indrel_change']=pd.Series(np.zeros(count,dtype=np.int8))
    df.loc[df['indrel']==99,'indrel_change'] = 1
    df["indrel"] = df["indrel"].astype(np.int8)
    df.drop('indrel',axis=1,inplace=True)

    # ind_actividad_cliente
    df.loc[df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = df["ind_actividad_cliente"].min()
    df['ind_actividad_cliente'] = df['ind_actividad_cliente'].astype(np.int8)

    # nomprovince
    df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
    df.loc[df.nomprov.isnull(),"nomprov"] = "UNKNOWN"

    # income
    incomes = df.loc[df.renta.notnull(),:].groupby("nomprov").agg({"renta":{"MedianIncome":np.median}})
    incomes.sort_values(by=("renta","MedianIncome"),inplace=True)
    incomes.reset_index(inplace=True)
    incomes.nomprov = incomes.nomprov.astype("category", categories=[i for i in df.nomprov.unique()],ordered=False)

    grouped = df.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
    new_incomes = pd.merge(df,grouped,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]
    new_incomes = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")
    df.sort_values("nomprov",inplace=True)
    df = df.reset_index()
    new_incomes = new_incomes.reset_index()

    df.loc[df.renta.isnull(),"renta"] = new_incomes.loc[df.renta.isnull(),"renta"].reset_index()
    df.loc[df.renta.isnull(),"renta"] = df.loc[df.renta.notnull(),"renta"].median()
    df.sort_values(by="fecha_dato",inplace=True)

    df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
    df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0

    string_data = df.select_dtypes(include=["object"])
    missing_columns = [col for col in string_data if string_data[col].isnull().any()]
    for col in missing_columns:
        print("Unique values for {0}:\n{1}\n".format(col,string_data[col].unique()))
    del string_data

    feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
    for col in feature_cols:
        df[col] = df[col].astype(np.int8)
    del df['index']

    df["year"] = pd.DatetimeIndex(df["fecha_dato"]).year
    df["mouth"] = pd.DatetimeIndex(df["fecha_dato"]).month
    df["year_fecha"] = pd.DatetimeIndex(df["fecha_alta"]).year
    df["mouth_fecha"] = pd.DatetimeIndex(df["fecha_alta"]).month
    df['fecha_diff'] = (df["year"] - df["year_fecha"]) * 12 + df["mouth"] - df["mouth_fecha"]
    df.drop(['fecha_dato','fecha_alta','year','year_fecha'],axis=1,inplace=True)

    print df.shape

    canal_entrada_label = ['KHE', 'KAT', 'KFC', 'KFA', 'KHK', 'KHD', 'KAS', 'KAG', 'RED',
                           'KAA', 'KAY', 'KAB', 'KHN', 'KHL', 'KCC', 'KAE', 'KBZ', 'KFD',
                           'KHM', 'KAI', 'KEY', 'KAW', 'KAF', 'KAR', '013', 'KAZ', 'KAH',
                           'KCI', 'KCH', 'KAJ']
    for item in canal_entrada_label:
        df[item] = pd.Series(np.zeros(count, dtype=np.int8))
        df.loc[df['canal_entrada'] == item, item] = 1

    nomprov_label = ['ALAVA','MADRID','MALAGA','MURCIA','NAVARRA','OURENSE','MELILLA','VALENCIA','UNKNOWN','TOLEDO',
                     'TERUEL','ZARAGOZA','ZAMORA','VALLADOLID','PONTEVEDRA','RIOJA, LA','SALAMANCA','PALMAS, LAS',
                     'PALENCIA','SEVILLA','TARRAGONA','SORIA','SEGOVIA','SANTA CRUZ DE TENERIFE','BARCELONA',
                    'BIZKAIA','BURGOS','CANTABRIA','CASTELLON','CADIZ','CIUDAD REAL','CORDOBA','CEUTA','CACERES',
                     'ASTURIAS','ALMERIA','ALICANTE','AVILA','BADAJOZ','ALBACETE','BALEARS, ILLES','CORUNA, A','CUENCA','GIRONA',
                     'GRANADA','GIPUZKOA','LUGO','LERIDA','LEON','GUADALAJARA','HUELVA','JAEN','HUESCA']
    for item in nomprov_label:
        df[item] = pd.Series(np.zeros(count, dtype=np.int8))
        df.loc[df['nomprov'] == item, item] = 1

    df.drop(['pais_residencia','canal_entrada','nomprov'],axis=1,inplace=True)
    print df.dtypes
    print df.shape
    df.to_csv('dataset/'+mouth_item+'_treated.csv',index=False)