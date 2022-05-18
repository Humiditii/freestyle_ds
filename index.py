from cgi import test
from cmath import inf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('Copy of Online Payment Fraud Detection.csv')

print(data.head())

class DataProcessing():
    """
        This a python class for handling Data and modelling stuffs
        varieties of methods available to call on the class's object to perform certain tasks
        arguments: data (Pd.DataFrame)
    """
    def __init__(self,df):
        self.df = df
    def heat_map(self):
        """
        This method print the heat map of the data showing null value magnitudes 
        """
        plt.figure(figsize=(40,40))
        sns.heatmap(self.df.isnull())
    def info(self):
        """
        this method prints
        """
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

    def onehot_encode(self, features):
        self.df = pd.get_dummies(self.df, prefix_sep='_', columns=features)
        print('OneHot done')

    def scaler(self, columns):
        sc = StandardScaler()
        for col in columns:
            self.df[col] = sc.fit_transform(np.array(self.df[col])).reshape(-1,1)
        print('Done scaling')
                 
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

    def split_data(self, train_thres=0.5):
        if train_thres >= 0.5 and train_thres <= 1.0:
            train_length = round(len(self.df) * train_thres)
            test_length = len(self.df) - train_length

            train = self.df[:train_length]
            test = self.df[train_length:]

            train_y = train['isFraud']
            train_x = train.drop('isFraud',1)

            self.train_y = train_y
            self.train_x = train_y

            test_y = test['isFraud']
            test_x = test.drop('isFraud',1)

            self.test_x = test_x
            self.test_y = test_y
        else:
            print('Invalid threshold')
    
    def data(self, data_type='train'):
        if data_type == 'test':
            return self.test_x,self.test_y
        if data_type == 'train':
            return self.train_x, self.train_y

    def trainer(self,x,y,model):
        for train_index, test_index in folds.split(x,y):
            start = time.time()
            x_train,x_test,y_train,y_test = x[train_index], x[test_index], y[train_index], y[test_index]
            score_gb.append(np.sqrt(train_score(gb, x_train,x_test,y_train,y_test)))
            score_xgb.append(np.sqrt(train_score(xgb_regressor, x_train,x_test,y_train,y_test)))
            score_dt.append(np.sqrt(train_score(dr, x_train,x_test,y_train,y_test)))
            score_rf.append(np.sqrt(train_score(rf, x_train,x_test,y_train,y_test)))
            score_stacked.append(np.sqrt(train_score(stacked, x_train,x_test,y_train,y_test)))
            prediction = xgb_regressor.predict(testing_data[fet].values)
            gb_pred = gb.predict(testing_data[fet].values)
            stackedPred = stacked.predict(testing_data[fet].values)
        pass
        
        

dp = DataProcessing(data)

dp.info()
#checking for null values
dp.heat_map()

print(dp.df.columns.to_list(), '\n')

dp.print_uniques('type') # categorical encoding

dp.print_uniques('step')

dp.corr()

print(dp.df['nameOrig'].value_counts()) # label encoding


print(dp.df['nameDest'].value_counts()) # label encoding

for col in dp.df.columns:
    if dp.df[col].dtypes == np.int64 or dp.df[col].dtypes == np.float64:
        print(dp.stat(col)) 

dp.df['step'] = 60*dp.df['step']

dp.df['sender_to_receiver'] = np.abs(dp.df['oldbalanceOrg'] - dp.df['newbalanceOrig'] ) / ( dp.df['newbalanceDest'] - dp.df['oldbalanceDest'] ) * dp.df['isFraud']

dp.df['step_amt_relation'] = (dp.df['step']/dp.df['step'].max()) * dp.df['amount']

le_encode_list = ['nameDest', 'nameOrig']

dp.encoder(le_encode_list)

dp.onehot_encode(['type'])

dp.df['sender_to_receiver'] = dp.df['sender_to_receiver'].fillna(0)

print(dp.df.head())

print(dp.df['sender_to_receiver'].info())

dp.split_data(0.6)

train_x, train_y = dp.data('train')

test_x, test_y = dp.data('test')


