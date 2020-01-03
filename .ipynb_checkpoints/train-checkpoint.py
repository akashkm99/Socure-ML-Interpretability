import torch
import torch, torch.nn as nn, torch.nn.functional as F
from load_data import get_dataset
from model import FraudNet


def train(model_name='fraud_net'):

    no_epochs = 10
    lr = 1e-3
    batch_size=64

    print ('Loading dataset ...')
    train_loader, valid_loader = get_dataset(minibatch_size=batch_size)

    net = FraudNet().cuda()
    print ('Model:')
    print (net)
    
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_valid_loss = float('inf')

    for i in range(no_epochs):

        for b, data in enumerate(train_loader):

            inputs, labels = data
            y_pred = net(inputs)
            loss = loss_fn(y_pred, labels)
            
            if b % 1000:
                print('Epochs: {}, batch: {} loss: {}'.format(i, b, loss))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        valid_loss = evaluate(valid_loader,net):
        print('Epochs: {}, Valid loss: {}'.format(i, valid_loss))

        if valid_loss < best_valid_loss:
            torch.save(net.state_dict(), './saved_checkpoints' + model_name + '.th')
            best_valid_loss = valid_loss
            print('Saving checkpoint at epoch: {}'.format(i))

def evaluate(test_loader, net):

    loss = 0.0
    steps = 0.0
    with torch.no_grad():
        loss_fn = nn.BCELoss()
        for b, data in enumerate(test_loader):

            inputs, labels = data
            y_pred = net(inputs)

            loss += loss_fn(y_pred, labels)
            steps += 1.0

    return loss/steps

def main():

    model_name = 'fraud_net'

    train(model_name=model_name)
    
    test_loader = get_dataset_test(minibatch_size=256)
    net = FraudNet().cuda()
    net.load_state_dict(torch.load('./saved_checkpoints' + model_name + '.th'))
    test_loss = evaluate(test_loader, net)    
    print('Test loss: {}'.format(test_loss))

if __name__ == "__main__":
    main()
