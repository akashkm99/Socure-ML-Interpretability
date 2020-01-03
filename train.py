import torch
import torch, torch.nn as nn, torch.nn.functional as F
from load_data import get_dataset, get_dataset_test
from model import FraudNet
import sys

def train(model_name='fraud_net'):

    no_epochs = 10
    lr = 1e-3
    batch_size=64
    weight_factor = 0.976    #weight given to class 1

    print ('Loading dataset ...')
    train_loader, valid_loader, test_loader = get_dataset(minibatch_size=batch_size)

    net = FraudNet().to(device='cuda')
    print ('Model:')
    print (net)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_valid_loss = float('inf')

    for i in range(no_epochs):

        for b, data in enumerate(train_loader):

            inputs, labels = data
            y_pred = net(inputs)
            mask = (labels > 0.5).float()
            batch_weights =  mask* weight_factor + (1-mask)*(1-weight_factor)
            loss = nn.BCELoss(weight=(batch_weights))(y_pred, labels)
            print ('batch_weights',batch_weights)
            
            if b % 1000:
                print('Epochs: {}, batch: {} loss: {}'.format(i, b, loss))
                sys.stdout.flush()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        valid_loss = evaluate(valid_loader,net)
        print('Epochs: {}, Valid loss: {}'.format(i, valid_loss))

        if valid_loss < best_valid_loss:
            test_loss = evaluate(test_loader,net)
            torch.save(net.state_dict(), './saved_checkpoints' + model_name + '_' + '{0:.3f}'.format(valid_loss) + '_' + '{0:.3f}'.format(test_loss) + '.th')
            best_valid_loss = valid_loss
            print('Saving checkpoint at epoch: {}'.format(i))

    print ("Test Loss: {}".format(test_loss))

def evaluate(test_loader, net):

    loss = 0.0
    steps = 0.0
    weight_factor = 0.976
    with torch.no_grad():
        for b, data in enumerate(test_loader):

            inputs, labels = data
            y_pred = net(inputs)

            mask = (labels > 0.5).float()
            batch_weights =  mask* weight_factor + (1-mask)*(1-weight_factor)
            loss += nn.BCELoss(weight=batch_weights)(y_pred, labels)
            steps += 1.0

    return loss/steps

def main():

    model_name = sys.argv[1]
    print ('cuda',torch.cuda.is_available())
    train(model_name=model_name)
    
#     test_loader = get_dataset_test(minibatch_size=256)
    net = FraudNet().to(device='cuda')
    net.load_state_dict(torch.load('./saved_checkpoints' + model_name + '.th'))
#     test_loss = evaluate(test_loader, net)    
#     print('Test loss: {}'.format(test_loss))

if __name__ == "__main__":
    main()
