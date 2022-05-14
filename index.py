from ast import Dict
from os import stat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('Copy of Online Payment Fraud Detection.csv')

print(data.head())

class DataProcessing():
    def __init__(self,df):
        self.df = df
    def heat_map(self):
        plt.figure(figsize=(40,40))
        sns.heatmap(self.df.isnull())
    def info(self):
        return self.df.info()
    def print_uniques(self, col_name):
        if type(col_name) == list:
            for col in col_name:
                print(self.df[col].unique())
        else:
            print(self.df[col_name].unique())
    def corr(self):
        plt.figure(figsize=(30,30))
        sns.heatmap(self.df.corr())
        plt.show()
    def encoder(self, columns):
        le = LabelEncoder()
        for col in columns:
            self.df[col] = le.fit_transform(self.df[col])
        print('Encoding done')

    def scaler(self, columns):
        sc = StandardScaler()
        for col in columns:
            self.df[col] = sc.fit_transform(np.array(self.df[col])).reshape(-1,1)

    def stat(self,column):

        stat_dict = {}
        stat_dict['max'] = 'nil'
        stat_dict['min'] = 'nil'
        stat_dict['mean'] = 'nil'
        stat_dict['median'] = 'nil'
        stat_dict['mode'] = 'nil'
        stat_dict['col_name'] = column

        stat_dict['mean'] = self.df[column].mean()
        stat_dict['median'] = self.df[column].median()
        stat_dict['mode'] = self.df[column].mode()[0]
        stat_dict['max'] = self.df[column].max()
        stat_dict['min'] = self.df[column].min()

        return stat_dict

dp = DataProcessing(data)

dp.info()
#checking for null values
#dp.heat_map()

print(dp.df.columns.to_list(), '\n')

#dp.print_uniques('type') # categorical encoding

#dp.print_uniques('step')

#dp.corr()

#print(dp.df['nameOrig'].value_counts()) # label encoding


#print(dp.df['nameDest'].value_counts()) # label encoding

for col in dp.df.columns:
    if dp.df[col].dtypes == np.int64 or dp.df[col].dtypes == np.float64:
        print(dp.stat(col)) 

dp.df['step'] = 60*dp.df['step']

dp.df['sender_to_receiver'] = (dp.df['oldbalanceOrg'] - dp.df['newbalanceOrig'] ) / ( dp.df['newbalanceDest'] - dp.df['oldbalanceDest'] ) * dp.df['isFraud']

dp.df['step_amt_relation'] = (dp.df['step']/dp.df['step'].max()) * dp.df['amount']

