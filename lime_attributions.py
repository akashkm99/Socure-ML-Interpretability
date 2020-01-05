import torch
import torch, torch.nn as nn, torch.nn.functional as F
from load_data import get_dataset, get_dataset_test
import sys
import numpy as np
from utils import load_fraudnet
import pickle
from lime.lime_tabular import LimeTabularExplainer
 
device='cuda'

def preprocess(grads):
    grads = np.abs(grads)/np.sum(grads,axis=1,keepdims=True)
    return grads

def preprocess1(embed_grad):

    embedding_sizes = np.array([10,5,3,2,3,20,2,2,2,2])
    split_indices = np.cumsum(embedding_sizes)

    categorical_grads = np.array([x.sum(1) for x in np.split(embed_grad,split_indices,axis=1)[:-1]]).swapaxes(0,1)
    numerical_grads = np.split(embed_grad,split_indices,axis=1)[-1]
    grads = np.concatenate([categorical_grads,numerical_grads],axis=1)
    grads = np.abs(grads)/np.sum(grads,axis=1,keepdims=True)

    return grads

def lime():
    
    print ('Loading dataset ...')
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_dataset(minibatch_size=32,sampling='None',numpy='True')

    net = load_fraudnet()
    lime_list = []
    explainer = LimeTabularExplainer(X_train,training_labels=Y_train)

    def func_call(x):
        print (x.shape)
        input_ = torch.from_numpy(x).to(device=device).float()
        prob_1 = net(input_).view(-1,1).cpu().data.numpy()
        prob_0 = 1-prob_1
        prob = np.concatenate([prob_0,prob_1],axis=1)
        return prob

    for i in range(X_test.shape[0]):

        exp = explainer.explain_instance(X_test[i,:], func_call, labels=(0,1), num_features=50)
        lime_list.append(exp)

    lime_list = np.array(lime_list)
    lime_list = preprocess(lime_list)
    pickle.dump(lime_list, open('./saved_attributions/lime.pkl', 'wb'))


def lime1():
    
    print ('Loading dataset ...')
    train_loader, valid_loader, test_loader = get_dataset(minibatch_size=1024,sampling='None')
    test_loader = get_dataset_test(minibatch_size=batch_size)

    net = load_fraudnet()

    train_embeds = []
    for data in train_loader:
        x,y = data
        embed = net.embedding_forward(x).cpu().data.numpy()
        train_embeds.append(embed)

    train_embeds = np.concatenate(train_embeds,axis=0)

    test_embeds = []
    for data in test_loader:
        x = data
        embed = net.embedding_forward(x).cpu().data.numpy()
        test_embeds.append(embed)

    test_embeds = np.concatenate(test_embeds,axis=0)

    print ('train test',train_embeds.shape,test_embeds.shape)

    lime_list = []
    explainer = LimeTabularExplainer(train_embeds)

    def func_call(x):
        print (x.shape)
        input_ = torch.from_numpy(x).to(device=device).float()
        prob_1 = net.feedforward(input_).view(-1,1).cpu().data.numpy()
        prob_0 = 1-prob_1
        prob = np.concatenate([prob_0,prob_1],axis=1)
        return prob

    for i in range(test_embeds.shape[0]):

        exp = explainer.explain_instance(test_embeds[i,:], func_call, labels=(0,1), num_features=91)
        lime_list.append(exp)

    lime_list = np.array(lime_list)
    lime_list = preprocess1(lime_list)
    pickle.dump(lime_list, open('./saved_attributions/lime.pkl', 'wb'))


if __name__ == "__main__":
    lime1()



