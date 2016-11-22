import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

mouth_list = ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28','2015-06-28','2015-07-28'
              ,'2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28','2016-01-28','2016-02-28'
              ,'2016-03-28','2016-04-28','2016-05-28']

for item in mouth_list:
    print item
    df = pd.read_csv('dataset/'+item+".csv",dtype={"sexo":str,"ind_nuevo":str,"ult_fec_cli_1t":str,
                                         "indext":str})

    df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
    df["fecha_alta"] = pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df.loc[df.age < 18,"age"] = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
    df.loc[df.age > 100,"age"] = df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True)
    df["age"].fillna(df["age"].mean(),inplace=True)
    df["age"] = df["age"].astype(int)

    df.loc[df["ind_nuevo"].isnull(),"ind_nuevo"] = 1

    df.loc[df.antiguedad.isnull(),"antiguedad"] = df.antiguedad.min()
    df.loc[df.antiguedad <0, "antiguedad"] = 0

    dates=df.loc[:,"fecha_alta"].sort_values().reset_index()
    median_date = int(np.median(dates.index.values))
    df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]

    df.loc[df.indrel.isnull(),"indrel"] = 1

    df.drop(["tipodom","cod_prov"],axis=1,inplace=True)

    df.loc[df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = \
        df["ind_actividad_cliente"].median()

    df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
    df.loc[df.nomprov.isnull(),"nomprov"] = "UNKNOWN"

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
    df.loc[df.conyuemp.isnull(), "conyuemp"] = 0

    string_data = df.select_dtypes(include=["object"])
    missing_columns = [col for col in string_data if string_data[col].isnull().any()]
    for col in missing_columns:
        print("Unique values for {0}:\n{1}\n".format(col,string_data[col].unique()))
    del string_data

    df.loc[df.indfall.isnull(),"indfall"] = "N"
    df.loc[df.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
    df.tiprel_1mes = df.tiprel_1mes.astype("category")

    # As suggested by @StephenSmith
    map_dict = { 1.0  : "1",
                 "1.0" : "1",
                 "1"   : "1",
                 "3.0" : "3",
                 "P"   : "P",
                 3.0   : "3",
                 2.0   : "2",
                 "3"   : "3",
                 "2.0" : "2",
                 "4.0" : "4",
                 "4"   : "4",
                 "2"   : "2"}

    df.indrel_1mes.fillna("P",inplace=True)
    df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
    df.indrel_1mes = df.indrel_1mes.astype("category")

    unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
    for col in unknown_cols:
        df.loc[df[col].isnull(),col] = "UNKNOWN"

    feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
    for col in feature_cols:
        df[col] = df[col].astype(int)
    del df['index']

    df["year"] = pd.DatetimeIndex(df["fecha_dato"]).year
    df["mouth"] = pd.DatetimeIndex(df["fecha_dato"]).month
    df["year_fecha"] = pd.DatetimeIndex(df["fecha_alta"]).year
    df["mouth_fecha"] = pd.DatetimeIndex(df["fecha_alta"]).month
    df['fecha_diff'] = (df["year"] - df["year_fecha"]) * 12 + df["mouth"] - df["mouth_fecha"]

    del df['fecha_dato']
    del df['fecha_alta']
    del df['year']
    del df['year_fecha']
    del df['ult_fec_cli_1t']

    df.to_csv('dataset/'+item+'_treated.csv',index=False)