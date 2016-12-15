import numpy as np
import pandas as pd

mouth_list = ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28','2015-06-28','2015-07-28'
              ,'2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28','2016-01-28','2016-02-28'
              ,'2016-03-28','2016-04-28','2016-05-28']
mouth_itr_list = ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28','2015-06-28','2015-07-28'
              ,'2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28','2016-01-28','2016-02-28'
              ,'2016-03-28','2016-04-28']

def status_change(x):
    diffs = x.diff().fillna(0)# first occurrence will be considered Maintained,
    #which is a little lazy. A better way would be to check if
    #the earliest date was the same as the earliest we have in the dataset
    #and consider those separately. Entries with earliest dates later than that have
    #joined and should be labeled as "Added"
    label = ["Added" if i==1 \
         else "Dropped" if i==-1 \
         else "Maintained" for i in diffs]
    return label

for item in mouth_itr_list:

    print 'this item ' + item
    df = pd.read_csv(item+"_treated.csv")
    # df = df[['ncodpers']]
    # print df.count()
    df = df[['ncodpers','ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1'
        ,'ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
             'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1'
        ,'ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1',
             'ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']]
    row_num = df['ncodpers'].count()
    print df['ncodpers'].count()
    print 'unique df '+str(len(df['ncodpers'].unique()))

    print 'compare with next item ' + mouth_list[mouth_list.index(item)+1]
    df_next = pd.read_csv(mouth_list[mouth_list.index(item)+1]+"_treated.csv")
    # df_next = df_next[['ncodpers']]
    # print df_next.count()
    df_next = df_next[['ncodpers','ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1'
        ,'ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
             'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1'
        ,'ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1',
             'ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']]
    print df_next['ncodpers'].count()
    print 'unique next df '+str(len(df_next['ncodpers'].unique()))

    df_change = pd.DataFrame(df.loc[df['ncodpers'].isin(df_next['ncodpers'])])
    print 'unique df_next in df  ' + str(len(df_change['ncodpers'].unique()))

    df_first_not_in_next = pd.DataFrame(df.loc[~df['ncodpers'].isin(df_next['ncodpers'])])
    print 'unique df not in next  ' + str(len(df_first_not_in_next['ncodpers'].unique()))

    df_next_not_in_first = pd.DataFrame(df_next.loc[~df_next['ncodpers'].isin(df['ncodpers'])])
    print 'unique df next in first  ' + str(len(df_next_not_in_first['ncodpers'].unique()))

    df_conct = pd.DataFrame(pd.concat([df,df_next,df_first_not_in_next,df_next_not_in_first]))
    print 'unique df  ' + str(len(df['ncodpers'].unique()))
    df_grouped = df_conct.groupby('ncodpers',sort=False)
    df = df_grouped.last()>df_grouped.first()
    df = df[0:row_num]
    df.to_csv(item+'_label.csv',sep=',')

    # df_change = pd.DataFrame(df.loc[df['ncodpers'].isin(df_next['ncodpers'])])
    # print 'unique df_next in df  ' + str(len(df_change['ncodpers'].unique()))
    #
    # df_diff = pd.DataFrame(df_next.loc[df_next['ncodpers'].isin(df_change['ncodpers'])])
    # print 'unique df in df_next ' + str(len(df_diff['ncodpers'].unique()))

    # label = np.empty((0,25),dtype=np.int64)
    # count =0
    # for row in df_change['ncodpers']:
    #     count+=1
    #     if count % 10000 == 0:
    #         print count
    #     row_label = np.array(df_diff.loc[df_diff['ncodpers']==row]) - np.array(df_change.loc[df_change['ncodpers']==row])
    #     row_label = np.append(row,row_label[0][1:])
    #     label = np.vstack((label,row_label))
    # np.savetxt(item+'_label.csv',label,delimiter=',')