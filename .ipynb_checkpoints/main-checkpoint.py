import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from load_data import get_dataset

class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 18)
        self.fc3 = nn.Linear(18, 20)
        self.fc4 = nn.Linear(20, 24)
        self.fc5 = nn.Linear(24, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.25)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x


def train():

    train_loader, test_loader = get_dataset()
    net = FraudNet().cuda().double()
    print (net)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for i in range(2):
        for b, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = net(inputs)
            loss = loss_fn(y_pred, labels)
            
            if b % 1000:
                print('Epochs: {}, batch: {} loss: {}'.format(i, b, loss))
            #reset gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()
