import torch
import torch, torch.nn as nn, torch.nn.functional as F
from load_data import get_dataset, get_dataset_test
import sys
from glob import glob
from metrics import calc_metrics_classification
import numpy as np
from tensorboardX import SummaryWriter
from RL_model import RLNet
from torch.distributions.bernoulli import Bernoulli
from utils import load_fraudnet
from imblearn.over_sampling import SMOTENC
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


default_features = np.zeros(50)
default_features[:10] = np.array([11.0,265.0,1.0,0.0,4.0,190.0,2.0,2.0,0.0,0.0]) #mode for categorical variables
default_features = torch.from_numpy(default_features).to(device='cuda').view(1,-1).float()


def smote_func(X,y):
    
    X = X.cpu().data.numpy() 
    y = y.cpu().data.numpy() 

    y_cat = (y>0.5).astype(np.int32)
    
    X = np.concatenate([X,np.expand_dims(y,1)],axis=1)

    smote_nc = SMOTENC(categorical_features=np.arange(0,10).tolist(), random_state=0, k_neighbors=3)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y_cat)
    new_x = X_resampled[:,:-1]
    new_y = X_resampled[:,-1]

    new_x = torch.from_numpy(new_x).to(device='cuda').float()
    new_y = torch.from_numpy(new_y).to(device='cuda').float()
    return new_x, new_y

def train(model_name='RL_net', tbd='logs', sparsity_lambda=0.5, ch=3):

    no_epochs = 30
    lr = 1e-3
    batch_size=1024

    print ('Loading dataset ...')
    train_loader, valid_loader, test_loader = get_dataset(minibatch_size=batch_size)
    fraud_net = load_fraudnet()
    net = RLNet(ch=ch).to(device='cuda').train()
    print ('Model:')
    print (net)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_valid_reward = -1*float('inf')
    writer = SummaryWriter('RL_runs/' + tbd)

    for i in range(no_epochs):

        for b, data in enumerate(train_loader):

            inputs, labels = data
            inputs, labels = smote_func(inputs, labels)

            y_logits = net(inputs)
            y_probs = F.sigmoid(y_logits)

            m = Bernoulli(probs=y_probs)
            selected_features = m.sample()
            number_selected = selected_features.sum(1) #fraction selected

            # print (selected_features,default_features)
            selected_inputs = inputs*selected_features + (1-selected_features)*default_features
            
            y_pred = fraud_net(selected_inputs)

            bce_loss = nn.BCELoss(reduction='none')(y_pred, labels)

            # print('number_selected, bce_loss',number_selected.shape,bce_loss.shape)
            number_dropped = 50 - number_selected
            reward = -1*bce_loss - sparsity_lambda*torch.abs(number_selected-5).unsqueeze(1)

            log_probs = m.log_prob(selected_features)
            print (log_probs.shape,reward.shape)
            loss = -log_probs * reward
            loss = loss.mean()

            
            if b % 100:
                print('Epochs: {}, batch: {}, loss: {}, reward: {}, bce loss: {}, fraction selected: {}'.format(i, b, loss, reward.mean().item(),bce_loss.mean().item(),number_selected.mean().item()))
                sys.stdout.flush()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('train_loss', loss, i)
        writer.add_scalar('reward', reward.mean(), i)
        writer.add_scalar('bce_loss', bce_loss.mean(), i)
        writer.add_scalar('number_selected', number_selected.mean(), i)

        valid_result = evaluate(valid_loader, fraud_net, net, sparsity_lambda)
        valid_result.update({'name':'Valid_Epoch_{}'.format(i)})
        print (valid_result)
        
        valid_reward = valid_result['reward']

        if valid_reward > best_valid_reward:
            torch.save(net.state_dict(), './saved_rl_checkpoints/' + model_name + '.th')
            best_valid_reward = valid_reward

            test_result = evaluate(test_loader, fraud_net, net, sparsity_lambda)
            test_result.update({'name':'test_Epoch_{}'.format(i)})
            print (test_result)

            print('Saving checkpoint at epoch: {}'.format(i))
    
    print (test_result)
            
def evaluate(test_loader, fraudnet, net, sparsity_lambda):

    total_loss = 0.0
    total_reward = 0.0
    total_bce_loss = 0.0
    total_number_selected = 0.0
    steps = 0.0
    net = net.eval()

    labels_list = []
    predicted_list = []
    loss_list = []
    
    with torch.no_grad():
        for b, data in enumerate(test_loader):

            inputs, labels = data
            inputs, labels = smote_func(inputs, labels)

            y_logits = net(inputs)
            y_probs = F.sigmoid(y_logits)

            m = Bernoulli(probs=y_probs)
            selected_features = m.sample()
            number_selected = selected_features.sum(1) #fraction selected

            selected_inputs = inputs*selected_features + (1-selected_features)*default_features
            y_pred = fraudnet(selected_inputs)

            bce_loss = nn.BCELoss(reduction='none')(y_pred, labels)

            number_dropped = 50 - number_selected
            reward = -1*bce_loss - sparsity_lambda*torch.abs(number_selected-5).unsqueeze(1)

            log_probs = m.log_prob(selected_features)
            loss = -log_probs * reward
            
            total_loss += loss.mean().item()
            total_bce_loss += bce_loss.mean().item()
            total_reward += reward.mean().item()
            total_number_selected += number_selected.mean().item()
            steps += 1

            predicted_list.append(y_pred.cpu().data.numpy())
            labels_list.append(labels.cpu().data.numpy())
            loss_list.append(bce_loss.cpu().data.numpy())

    predicted_list = np.array([x for y in predicted_list for x in y])
    labels_list = np.array([x for y in labels_list for x in y])
    actual_labels = (labels_list >=0.5).astype(np.int32)
    loss_list = np.array([x for y in loss_list for x in y])    
    positive_loss = loss_list[labels_list >= 0.5].mean()
    negative_loss = loss_list[labels_list < 0.5].mean()
    overall_loss = loss_list.mean()

    result = calc_metrics_classification(actual_labels,predicted_list)
    result.update({'positive_bce_loss':positive_loss,'negative_bce_loss':negative_loss,'overall_bce_loss':overall_loss})

    total_loss /= steps    
    total_reward /= steps    
    total_number_selected  /= steps    
    result.update({'loss':total_loss,'reward':total_reward,'number_selected':total_number_selected})
    return result

def preprocessData(df):
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

def testLoad():
    data = pd.read_csv('./dataset/competition-data/test.csv')
    X = data.drop(columns="id")
    X = preprocessData(X).values
    sc = StandardScaler()
    
    X[:, 10:] = sc.fit_transform(X[:, 10:]) # normalize numeric data
    
    
    X_train = torch.from_numpy(X).float().to(device='cuda')
    return X_train

def testing(path, sparsity_lambda, ch):
    fraud_net = load_fraudnet()
    net = RLNet(ch).to(device='cuda')
    net = net.eval()
    net.load_state_dict(torch.load(path))
    net = net.eval()
    x_test = testLoad()
    attribute_list = []
    
    with torch.no_grad():
        y_logits = net(x_test)
    y_probs = F.sigmoid(y_logits).detach().cpu().numpy()
    rl_attributes = y_probs/y_probs.sum(axis=1)[:, None]
    print (rl_attributes)
        
#         for b, data in enumerate(test_loader):
#             inputs, labels = data
#             y_logits = net(inputs)
#             y_probs = F.sigmoid(y_logits).detach().numpy()
#             y_probs = y_probs/y_probs.sum(axis=1)[:, None]
#             attribute_list.append(y_probs)
#     rl_attributes = np.concatenate(y_probs,axis=0)
    db = {}
    db['attributes'] = rl_attributes
    dbfile = open('RL_fc_0.3_0.02_3_attributes', 'ab') 
    pickle.dump(db, dbfile)
    dbfile.close()
    
def main():
    test=True
    if test:
        testing('./saved_rl_checkpoints/RL_fc_0.3_0.02_3.th', 0.02,3)
    else:
        model_name = sys.argv[1]
        sparsity_lambda = 0.02 #float(sys.argv[2]) # number of layers
        ch = int(sys.argv[2])
        tbd = sys.argv[3]
        print ('cuda',torch.cuda.is_available())
        train(model_name=model_name, tbd=tbd, sparsity_lambda=sparsity_lambda, ch=ch)
        # test(sys.argv[1],int(sys.argv[2]))

if __name__ == "__main__":
    main()



