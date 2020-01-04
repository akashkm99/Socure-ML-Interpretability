import torch
import torch, torch.nn as nn, torch.nn.functional as F
from load_data import get_dataset, get_dataset_test
from model import FraudNet
import sys
from glob import glob
from metrics import calc_metrics_classification
import numpy as np
from tensorboardX import SummaryWriter



def smote(X,y):
    
    X = X.cpu().data.numpy() 
    y = y.cpu().data.numpy() 

    y_cat = (y>0.5).astype(np.int32)
    
    X = np.concatenate([X,np.expand_dims(y,1)],axis=1)

    smote_nc = SMOTENC(categorical_features=np.arange(0,10).tolist(), random_state=0)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y_cat)
    new_x = X_resampled[:,:-1]
    new_y = X_resampled[:,-1]

    new_x = torch.from_numpy(new_x).to(device=device).float()
    new_y = torch.from_numpy(new_y).to(device=device).float()
    return new_x, new_y



def train(model_name='fraud_net', ch=1, tbd='logs', smote=True):

    no_epochs = 30
    lr = 1e-3
    batch_size=256
    weight_factor = 1.0#40.67    #weight given to class 1

    print ('Loading dataset ...')
    train_loader, valid_loader, test_loader = get_dataset(minibatch_size=batch_size)

    net = FraudNet(ch).to(device='cuda').train()
    print ('Model:')
    print (net)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_valid_loss = float('inf')
    writer = SummaryWriter('runs/' + tbd)

    for i in range(no_epochs):

        for b, data in enumerate(train_loader):

            inputs, labels = data

            if smote:
                inputs, labels = smote(inputs, labels)

            y_pred = net(inputs)
            mask = (labels > 0.5).float()
            batch_weights =  mask* weight_factor + (1-mask)*(1)
            loss = nn.BCELoss(weight=(batch_weights))(y_pred, labels)
            # print ('batch_weights',batch_weights)
            
            if b % 100000000:
                print('Epochs: {}, batch: {} loss: {}'.format(i, b, loss))
                sys.stdout.flush()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar('train_loss', loss, i)
        valid_loss = evaluate(valid_loader,net)
        writer.add_scalar('valid_loss', valid_loss, i)
        test_loss = evaluate(test_loader,net)
        writer.add_scalar('test_loss', test_loss, i)
        
        print('Epochs: {}, Valid loss: {}, Test Loss: {}'.format(i, valid_loss, test_loss))

        if valid_loss < best_valid_loss:
            torch.save(net.state_dict(), './saved_checkpoints/' + model_name + '_' + '{0:.3f}'.format(valid_loss) + '_' + '{0:.3f}'.format(test_loss) + '.th')
            best_valid_loss = valid_loss
            print('Saving checkpoint at epoch: {}'.format(i))
            
def evaluate(test_loader, net):

    loss = 0.0
    steps = 0.0
    weight_factor = 1.0#40.67
    net = net.eval()
    with torch.no_grad():
        for b, data in enumerate(test_loader):

            inputs, labels = data
            y_pred = net(inputs)

            mask = (labels > 0.5).float()
            batch_weights =  mask* weight_factor + (1-mask)*(1)
            loss += nn.BCELoss(weight=batch_weights)(y_pred, labels)
            steps += 1.0

    return loss/steps


def test(model_name, ch):

    batch_size=64
    train_loader, valid_loader, test_loader = get_dataset(minibatch_size=batch_size)

    net = FraudNet(ch).to(device='cuda')
    path = glob('./saved_checkpoints/' + model_name + '_*')[0]
    net.load_state_dict(torch.load(path))

    loss = 0.0
    steps = 0.0
    weight_factor = 1.0
    net = net.eval()

    predicted_list = []
    labels_list = []
    loss_list = []

    with torch.no_grad():
        for b, data in enumerate(test_loader):

            inputs, labels = data
            y_pred = net(inputs)

            mask = (labels > 0.5).float()
            batch_weights =  mask* weight_factor + (1-mask)*(1)
            loss += nn.BCELoss(weight=batch_weights)(y_pred, labels)
            steps += 1.0

            bce_loss = nn.BCELoss(reduction='none')(y_pred, labels)

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
    weighted_loss = loss/steps

    result = calc_metrics_classification(actual_labels,predicted_list)
    result.update({'positive_loss':positive_loss,'negative_loss':negative_loss,'overall_loss':overall_loss,'weighted_loss':weighted_loss})
    print(result)

def main():

    model_name = sys.argv[1]
    ch = int(sys.argv[2]) # number of layers
    tbd = sys.argv[3]
    print ('cuda',torch.cuda.is_available())
    train(model_name=model_name, ch=ch, tbd=tbd)
    test(sys.argv[1],int(sys.argv[2]))

#     test_loader = get_dataset_test(minibatch_size=256)
    # net = FraudNet().to(device='cuda')
    # net.load_state_dict(torch.load('./saved_checkpoints' + model_name + '.th'))
#     test_loss = evaluate(test_loader, net)    
#     print('Test loss: {}'.format(test_loss))

if __name__ == "__main__":
    main()
