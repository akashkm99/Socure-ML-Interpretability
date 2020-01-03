import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.no_categorical = 10
        self.no_numerical = 40

        self.embedding_input_sizes = [501, 264, 60, 61, 107, 11624, 4, 3, 104, 5] # number of classes
        self.embedding_output_sizes = [10,5,3,2,3,20,2,2,2,2] # output embedding shape

        embed_list = []
        for i in range(self.no_categorical):
            embed_list.append(nn.Embedding(self.embedding_input_sizes[i],self.embedding_output_sizes[i]))

        self.embedding_layers = nn.ModuleList(embed_list)

        self.input_size = np.array(self.embedding_output_sizes).sum() + self.no_numerical

        self.fc1 = nn.Linear(self.input_size, 1)
#         self.fc2 = nn.Linear(16, 18)
#         self.fc3 = nn.Linear(18, 20)
#         self.fc4 = nn.Linear(20, 24)
#         self.fc5 = nn.Linear(24, 1)

    def forward(self, x):

        categorical_embeddings = []

        for i in range(self.no_categorical):
            embed = self.embedding_layers[i](x[:,i])  #(B,embed_output_size)
            categorical_embeddings.append(embed)

        categorical_embeddings = torch.cat(categorical_embeddings,dim=1) # (B, sum of embed_output_size)
        numerical_features = x[:,self.no_categorical:]
        
        x = torch.cat([categorical_embeddings,numerical_features],dim=1) #(B, sum of embed_output_size + no. of numerical_features)

        x = F.sigmoid(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x, p=0.25)
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = F.sigmoid(self.fc5(x))

        return x
