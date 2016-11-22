# # Column Name	Description
# fecha_dato	The table is partitioned for this column
# ncodpers	Customer code
# ind_empleado	Employee index: A active, B ex employed, F filial, N not employee, P pasive
# pais_residencia	Customer's Country residence
# sexo	Customer's sex
# age	Age
# fecha_alta	The date in which the customer became as the first holder of a contract in the bank
# ind_nuevo	New customer Index. 1 if the customer registered in the last 6 months.
# antiguedad	Customer seniority (in months)
# indrel	1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)
# ult_fec_cli_1t	Last date as primary customer (if he isn't at the end of the month)
# indrel_1mes	Customer type at the beginning of the month ,1 (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner)
# tiprel_1mes	Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
# indresi	Residence index (S (Yes) or N (No) if the residence country is the same than the bank country)
# indext	Foreigner index (S (Yes) or N (No) if the customer's birth country is different than the bank country)
# conyuemp	Spouse index. 1 if the customer is spouse of an employee
# canal_entrada	channel used by the customer to join
# indfall	Deceased index. N/S
# tipodom	Addres type. 1, primary address
# cod_prov	Province code (customer's address)
# nomprov	Province name
# ind_actividad_cliente	Activity index (1, active customer; 0, inactive customer)
# renta	Gross income of the household
# segmento	segmentation: 01 - VIP, 02 - Individuals 03 - college graduated
# ind_ahor_fin_ult1	Saving Account
# ind_aval_fin_ult1	Guarantees
# ind_cco_fin_ult1	Current Accounts
# ind_cder_fin_ult1	Derivada Account
# ind_cno_fin_ult1	Payroll Account
# ind_ctju_fin_ult1	Junior Account
# ind_ctma_fin_ult1	Mas particular Account
# ind_ctop_fin_ult1	particular Account
# ind_ctpp_fin_ult1	particular Plus Account
# ind_deco_fin_ult1	Short-term deposits
# ind_deme_fin_ult1	Medium-term deposits
# ind_dela_fin_ult1	Long-term deposits
# ind_ecue_fin_ult1	e-account
# ind_fond_fin_ult1	Funds
# ind_hip_fin_ult1	Mortgage
# ind_plan_fin_ult1	Pensions
# ind_pres_fin_ult1	Loans
# ind_reca_fin_ult1	Taxes
# ind_tjcr_fin_ult1	Credit Card
# ind_valo_fin_ult1	Securities
# ind_viv_fin_ult1	Home Account
# ind_nomina_ult1	Payroll
# ind_nom_pens_ult1	Pensions
# ind_recibo_ult1	Direct Debit

import pandas as pd
import numpy as np

mouth_list = ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28','2015-06-28','2015-07-28'
              ,'2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28','2016-01-28','2016-02-28'
              ,'2016-03-28','2016-04-28','2016-05-28']

for item in mouth_list:
    print item
    df = pd.read_csv(item+"_treated.csv",engine='c',nrows=10)
    df["year"] = pd.DatetimeIndex(df["fecha_dato"]).year
    df["mouth"] = pd.DatetimeIndex(df["fecha_dato"]).month
    df["day"] = pd.DatetimeIndex(df["fecha_dato"]).day
    df["year_fecha"] = pd.DatetimeIndex(df["fecha_alta"]).year
    df["mouth_fecha"] = pd.DatetimeIndex(df["fecha_alta"]).month
    df["day_fecha"] = pd.DatetimeIndex(df["fecha_alta"]).day
    df['year_diff'] = df["year"] - df["year_fecha"]
    df['mouth_diff'] = df['year_diff']*12 + df["mouth"] - df["mouth_fecha"]
    del df['fecha_dato']
    del df['fecha_alta']
    del df['ncodpers']

    print df.dtypes


#!/usr/bin/python

def loadfmap( fname ):
    fmap = {}
    nmap = {}

    for l in open( fname ):
        arr = l.split()
        if arr[0].find('.') != -1:
            idx = int( arr[0].strip('.') )
            assert idx not in fmap
            fmap[ idx ] = {}
            ftype = arr[1].strip(':')
            content = arr[2]
        else:
            content = arr[0]
        for it in content.split(','):
            if it.strip() == '':
                continue
            k , v = it.split('=')
            fmap[ idx ][ v ] = len(nmap) + 1
            nmap[ len(nmap) ] = ftype+'='+k
    return fmap, nmap

def write_nmap( fo, nmap ):
    for i in range( len(nmap) ):
        fo.write('%d\t%s\ti\n' % (i, nmap[i]) )

# start here
fmap, nmap = loadfmap( 'agaricus-lepiota.fmap' )
fo = open( 'featmap.txt', 'w' )
write_nmap( fo, nmap )
fo.close()

fo = open( 'agaricus.txt', 'w' )
for l in open( 'agaricus-lepiota.data' ):
    arr = l.split(',')
    if arr[0] == 'p':
        fo.write('1')
    else:
        assert arr[0] == 'e'
        fo.write('0')
    for i in range( 1,len(arr) ):
        fo.write( ' %d:1' % fmap[i][arr[i].strip()] )
    fo.write('\n')

fo.close()
