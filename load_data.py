import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer

from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
import numpy as np


def preprocess(df):
    # x is pandas dataframe
    for i in range(1,11):
        # first 10 categorical variables
        k="C"+str(i)
        df[k].fillna(df[k].mode()[0], inplace=True) # fill with mode
        df[k] = df[k] - df[k].min()
    # fill with median in case of continuous
    imputer = Imputer(missing_values=np.nan, strategy='median', axis=0)
    for i in range(1,41):
        k="V"+str(i)
        df[k] = pd.to_numeric(df[k], errors='coerce')
        df[k].fillna(df[k].median(), inplace=True)
    return df

def get_dataset(minibatch_size=64):

    data = pd.read_csv('./dataset/competition-data/train.csv')

    """
    data.head()
    print(data.shape)
    print(data.columns)
    data.isnull().sum().any()  #check for n/a
    """

    y = data['model_score'].values # extracting labels
    
    X = data.drop(columns="model_score")
    X = preprocess(X).values
    sc = StandardScaler()
    
    X[:, 10:] = sc.fit_transform(X[:, 10:]) # normalize numeric data

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    X_train = torch.from_numpy(X_train).cuda()
    Y_train = torch.from_numpy(Y_train).cuda().float()
    train = data_utils.TensorDataset(X_train, Y_train)
    train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True)

    X_test = torch.from_numpy(X_test).cuda()
    Y_test = torch.from_numpy(Y_test).cuda().float()
    test = data_utils.TensorDataset(X_test, Y_test)
    test_loader = data_utils.DataLoader(test, batch_size=minibatch_size, shuffle=True)

    return train_loader, test_loader

def get_dataset_test(minibatch_size=64):

    data = pd.read_csv('./dataset/competition-data/test.csv')

    """
    data.head()
    print(data.shape)
    print(data.columns)
    data.isnull().sum().any()  #check for n/a
    """

    y = data['model_score'].values # extracting labels
    
    X = data.drop(columns="model_score")
    X = preprocess(X).values
    sc = StandardScaler()
    
    X[:, 10:] = sc.fit_transform(X[:, 10:]) # normalize numeric data
    
    
    X_train = torch.from_numpy(X).cuda()
    Y_train = torch.from_numpy(y).cuda().float()
    train = data_utils.TensorDataset(X, y)
    train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True)

    return train_loader



if __name__ == '__main__':
    t,v = get_dataset()
    
    