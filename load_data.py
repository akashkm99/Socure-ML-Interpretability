import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
import numpy as np
from imblearn.over_sampling import SMOTENC
from collections import Counter


def smote(X,y):
    
    y_cat = (y>0.5).astype(np.int32)
    
    X = np.concatenate([X,np.expand_dims(y,1)],axis=1)

    smote_nc = SMOTENC(categorical_features=np.arange(0,10).tolist(), random_state=0)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y_cat)
    new_x = X_resampled[:,:-1]
    new_y = X_resampled[:,-1]
    return new_x, new_y

def preprocess(df):
    # x is pandas dataframe
    for i in range(1,11):
        # first 10 categorical variables
        k="C"+str(i)
        df[k].fillna(df[k].mode()[0], inplace=True) # fill with mode
        df[k] = df[k] - df[k].min()
    # fill with median in case of continuous
#     imputer = Imputer(missing_values=np.nan, strategy='median', axis=0)
    for i in range(1,41):
        k="V"+str(i)
        df[k] = pd.to_numeric(df[k], errors='coerce')
        df[k].fillna(df[k].median(), inplace=True)
    return df

def get_dataset(minibatch_size=64, sampling='repeat', device='cuda', numpy=False):

    data = pd.read_csv('./dataset/competition-data/train.csv')
    
    if sampling == 'repeat':
        norm_1 = pd.concat([data[data['model_score']>0.5]]*41)
        norm_0 = data[data['model_score']<0.5]
        data = pd.concat([norm_1, norm_0])
    
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

    # X = sc.fit_transform(X) # normalize numeric data
    X[:, 10:] = sc.fit_transform(X[:, 10:]) # normalize numeric data

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=1)
    
    if sampling == 'smote':
        print(X_train.shape)    
        X_train, Y_train = smote(X_train, Y_train)
        np.savetxt('dataset/SMOTE_train_X.txt', X_train, delimiter=',')
        np.savetxt('dataset/SMOTE_train_Y.txt', Y_train, delimiter=',')
        print(X_train.shape)
        return 
    
    if numpy:
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
        
    X_train = torch.from_numpy(X_train).to(device=device).float()
    Y_train = torch.from_numpy(Y_train).to(device=device).float()
    train = data_utils.TensorDataset(X_train, Y_train)
    train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True)

    X_val = torch.from_numpy(X_val).to(device=device).float()
    Y_val = torch.from_numpy(Y_val).to(device=device).float()
    val = data_utils.TensorDataset(X_val, Y_val)
    val_loader = data_utils.DataLoader(val, batch_size=minibatch_size, shuffle=True)

    X_test = torch.from_numpy(X_test).to(device=device).float()
    Y_test = torch.from_numpy(Y_test).to(device=device).float()
    test = data_utils.TensorDataset(X_test, Y_test)
    test_loader = data_utils.DataLoader(test, batch_size=minibatch_size, shuffle=True)

    return train_loader, val_loader, test_loader
    
def get_dataset_test(minibatch_size=64):

    data = pd.read_csv('./dataset/competition-data/test.csv')

    """
    data.head()
    print(data.shape)
    print(data.columns)
    data.isnull().sum().any()  #check for n/a
    """

    X = data.drop(columns="id")
    X = preprocess(X).values
    sc = StandardScaler()
    
    X[:, 10:] = sc.fit_transform(X[:, 10:]) # normalize numeric data
    
    
    X_train = torch.from_numpy(X).to(device='cuda').float()
    train = data_utils.TensorDataset(X_train)
    train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True)

    return train_loader

if __name__ == '__main__':
    t,v = get_dataset(sampling='smote')
    
    
