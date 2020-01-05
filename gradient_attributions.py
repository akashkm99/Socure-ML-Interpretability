import torch
import torch, torch.nn as nn, torch.nn.functional as F
from load_data import get_dataset, get_dataset_test
import sys
import numpy as np
from utils import load_fraudnet
import pickle
import lime.lime_tabular

device='cuda'

def preprocess(embed_grad):

    embedding_sizes = np.array([10,5,3,2,3,20,2,2,2,2])
    split_indices = np.cumsum(embedding_sizes)

    categorical_grads = np.array([x.sum(1) for x in np.split(embed_grad,split_indices,axis=1)[:-1]]).swapaxes(0,1)
    numerical_grads = np.split(embed_grad,split_indices,axis=1)[-1]
    grads = np.concatenate([categorical_grads,numerical_grads],axis=1)
    grads = np.abs(grads)
    grads = grads/np.sum(grads,axis=1,keepdims=True)
    return grads

def gradients():

    print ('Loading dataset ...')
    batch_size = 1024
    # train_loader, valid_loader, test_loader = get_dataset(minibatch_size=batch_size, sampling='None')
    test_loader = get_dataset_test(minibatch_size=batch_size)
    net = load_fraudnet()
    gradient_list = []

    for b, data in enumerate(test_loader):

        inputs = data[0]
        y_pred = net(inputs)
        y_pred.sum().backward()

        embed_grad = net.embedding.grad.cpu().data.numpy()
        gradient_list.append(embed_grad)

    gradient_list = np.array([x for y in gradient_list for x in y])
    gradient_list = preprocess(gradient_list)
    print ('grads shape',gradient_list.shape)
    pickle.dump(gradient_list, open('./saved_attributions/gradients.pkl', 'wb'))

def integrated_gradients(steps=50):
    
    print ('Loading dataset ...')
    batch_size = 4
    # train_loader, valid_loader, test_loader = get_dataset(minibatch_size=batch_size, sampling='None')
    test_loader = get_dataset_test(minibatch_size=batch_size)

    net = load_fraudnet()
    int_gradient_list = []

    baseline = np.zeros(50)
    baseline[:10] = np.array([11.0,265.0,1.0,0.0,4.0,190.0,2.0,2.0,0.0,0.0]) #mode for categorical variables

    baseline = torch.from_numpy(baseline).view(1,-1).to(device).float()
    baseline_embeddings = net.embedding_forward(baseline) #[1,91]

    for b, data in enumerate(test_loader):

        inputs = data[0]  #inputs - [B,50]
        embeddings = net.embedding_forward(inputs)  #[B,91]

        baseline_embeddings = baseline_embeddings.detach()

        diff = (embeddings - baseline_embeddings) #[B,91]

        incr = diff.unsqueeze(2)*torch.linspace(0,1,steps).view(1,1,-1).to(device) #[B,91,steps]

        new_embeddings = baseline_embeddings.unsqueeze(2) + incr
        new_embeddings = new_embeddings.transpose(1,2).contiguous().view(-1,91)  #[B*steps,91]
        new_embeddings.retain_grad()

        y_predicted = net.feedforward(new_embeddings)  #[B*steps,1] 
        y_predictions = y_predicted.view(-1,steps)
        y_predicted.sum().backward()  

        gradients = new_embeddings.grad.view(-1,steps,91) #[B,steps,91]
        trapezoid_sum = torch.mean((gradients[:,:-1,:] + gradients[:,1:,:])/2.0, dim=1) #[B,91]
        int_grad =  (trapezoid_sum*diff).cpu().data.numpy() #[B,91]

        ###sanity check 
        # print('int_grad sum, output diff', int_grad.sum(1), y_predictions[:,-1] - y_predictions[:,0])

        int_gradient_list.append(int_grad)

    int_gradient_list = np.array([x for y in int_gradient_list for x in y])
    int_gradient_list = preprocess(int_gradient_list)
    print ('int_grads shape',int_gradient_list.shape)
    pickle.dump(int_gradient_list, open('./saved_attributions/integrated_gradients.pkl', 'wb'))


if __name__ == "__main__":
    gradients()
    integrated_gradients(steps=1024)




