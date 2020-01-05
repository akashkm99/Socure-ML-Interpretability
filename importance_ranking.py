import torch
import torch, torch.nn as nn, torch.nn.functional as F
from load_data import get_dataset, get_dataset_test
import sys
import numpy as np
from utils import load_fraudnet
import pickle
import pandas as pd 
import numpy as np

baseline = np.zeros(50)
baseline[:10] = np.array([11.0,265.0,1.0,0.0,4.0,190.0,2.0,2.0,0.0,0.0]) #mode for categorical variables

def minimum_length(inputs,net,ranking):

    inputs = inputs.repeat(51,1)
    for i, idx in enumerate(ranking):
        inputs[i+1:,idx] = baseline[idx]
    
    y_pred = (net(inputs) > 0.5).long().cpu().data.numpy().squeeze()

    y_original = y_pred[0]
    for i in range(1,51):
        if (y_pred[i] == y_original):
            return y_original,i
    return y_original,50


def importance_rank(name):

    print ('Loading dataset ...')
    batch_size = 1
    test_loader = get_dataset_test(minibatch_size=batch_size)
    net = load_fraudnet()
    print (name)
    rank_list = []

    gradients = pickle.load(open('saved_attributions/'+name, 'rb'))
    gradient_length = []
    y_original = []
    if name in ['shap','rl','deeplift']:
        gradients = gradients['attributes']
    gradients = np.array(gradients)
    print (gradients)
    for idx, data in enumerate(test_loader):

        inputs = data[0]
        gradients_ranking = np.argsort(np.array(gradients[idx]))[::-1]    
        y,l = minimum_length(inputs,net,gradients_ranking)
        gradient_length.append(l)
        y_original.append(y)

    result_dict = {'predicted':y_original,'length':gradient_length}
    df = pd.DataFrame(result_dict)
    print (df)
    pickle.dump(df, open('./saved_importance_rank/'+name+'.pkl', 'wb'))


if __name__ == "__main__":
    names = ['shap','rl','deeplift','gradients','integrated_gradients']
    for name in names:
        importance_rank(name)