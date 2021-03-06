# from https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb

from GRU_Cell import GRUCell

import torch
import torch.nn as nn
from torch.autograd import Variable

import math

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    
    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        outs = []
        
        hn = h0[0,:,:]
        
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:], hn)
            outs.append(hn)
        
        out = outs[-1].squeeze()
        
        out = self.fc(out)
        # out.size() --> 100, 10
        return out
