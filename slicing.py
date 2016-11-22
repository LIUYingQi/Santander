import csv
import numpy as np
import os

mouth_list = ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28','2015-06-28','2015-07-28'
              ,'2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28','2016-01-28','2016-02-28'
              ,'2016-03-28','2016-04-28','2016-05-28']

num_list = [0,625457,1252851,1882060,2512427,3144384,3776494,4606311,5449512,6314952,7207203,8113312,9025333
            ,9941602,10862506,11787582,12715856,13647309]

train_file = 'train_ver2.csv'

with open(train_file,'rb') as train_file:
    train_file_reader = csv.reader(train_file)
    title = train_file_reader.next()
    for i in range(len(mouth_list)):
        with open(mouth_list[i]+'.csv','ab') as file:
            print 'write :' + str(i)
            writer = csv.writer(file,delimiter=',')
            writer.writerow(title)
            for item_num in range(num_list[i+1]-num_list[i]):
                writer.writerow(train_file_reader.next())