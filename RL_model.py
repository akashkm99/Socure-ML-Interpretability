import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class RLNet(nn.Module):
    def __init__(self, ch=3, continuous=False):
        super().__init__()
        self.continuous = continuous

        if continuous:
            self.no_categorical = 0
            self.no_numerical = 50
            self.input_size = self.no_numerical 

        else:
            self.no_categorical = 10
            self.no_numerical = 40
            self.embedding_input_sizes = [502, 266, 60, 61, 109, 11625, 4, 3, 106, 5] # number of classes
            self.embedding_output_sizes = [10,5,3,2,3,20,2,2,2,2] # output embedding shape
            embed_list = []
            for i in range(self.no_categorical):
                embed_list.append(nn.Embedding(self.embedding_input_sizes[i],self.embedding_output_sizes[i]))

            self.embedding_layers = nn.ModuleList(embed_list)
            self.input_size = np.array(self.embedding_output_sizes).sum() + self.no_numerical

        self.ch = ch
        self.output_size = self.no_categorical + self.no_numerical
        if self.ch==3:
            self.fc1 = nn.Linear(self.input_size, self.input_size//2)
            self.fc2 = nn.Linear(self.input_size//2, self.input_size//2)
            self.fc3 = nn.Linear(self.input_size//2, self.output_size)
        elif self.ch==4:
            self.fc1 = nn.Linear(self.input_size, self.input_size//2)
            self.fc2 = nn.Linear(self.input_size//2, self.input_size//2)
            self.fc3 = nn.Linear(self.input_size//2, self.input_size//4)
            self.fc4 = nn.Linear(self.input_size//4, self.output_size)
        elif self.ch==5:
            self.fc1 = nn.Linear(self.input_size, self.input_size//2)
            self.fc2 = nn.Linear(self.input_size//2, self.input_size//2)
            self.fc3 = nn.Linear(self.input_size//2, self.input_size//4)
            self.fc4 = nn.Linear(self.input_size//4, self.input_size//4)
            self.fc5 = nn.Linear(self.input_size//4, self.output_size)
        elif self.ch==6:
            self.fc1 = nn.Linear(self.input_size, self.input_size//2)
            self.fc2 = nn.Linear(self.input_size//2, self.input_size//2)
            self.fc3 = nn.Linear(self.input_size//2, self.input_size//4)
            self.fc4 = nn.Linear(self.input_size//4, self.input_size//4)
            self.fc5 = nn.Linear(self.input_size//4, self.input_size//2)
            self.fc6 = nn.Linear(self.input_size//2, self.output_size)
    
    def embedding_forward(self,x):

        categorical_embeddings = []

        categorical_features = x[:,:self.no_categorical].long()
        #print (categorical_features)
        for i in range(self.no_categorical):
            #print ('i',i)
            embed = self.embedding_layers[i](categorical_features[:,i])  #(B,embed_output_size)
            categorical_embeddings.append(embed)

        categorical_embeddings = torch.cat(categorical_embeddings,dim=1) # (B, sum of embed_output_size)
        numerical_features = x[:,self.no_categorical:]
        x = torch.cat([categorical_embeddings,numerical_features],dim=1) #(B, sum of embed_output_size + no. of numerical_features)
        return x
    
    def feedforward(self,x):
        p=0.3         
        if self.ch==3:
            x = F.relu(self.fc1(x))
            # x = F.dropout(x, p=p)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=p)
            x = self.fc3(x)
        elif self.ch==4:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=p)
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
        elif self.ch==5:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=p)
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = self.fc5(x)
        elif self.ch==6:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=p)
            x = F.relu(self.fc3(x))
            x = F.dropout(x, p=p)
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = self.fc6(x)
        return x

    def forward(self, x):
        if not self.continuous:
            x = self.embedding_forward(x)
        
        x = self.feedforward(x)
        return x