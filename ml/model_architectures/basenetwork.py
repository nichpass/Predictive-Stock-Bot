from torch import mean, std
import torch.nn as nn

class BaseNetwork(nn.Module):
    
    def __init__(self, inputs=6):
        super(BaseNetwork, self).__init__()

        h1_size = 15
        h2_size = 10

        self.fc1 = nn.Linear(inputs, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(p=0.3) # deal with this guy later when results matter
        self.dropout2 = nn.Dropout(p=0.3)
        
    # x represents our data
    def forward(self, x):
        x = self.fc1(x)
        # x = self.dropout1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = self.dropout1(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x