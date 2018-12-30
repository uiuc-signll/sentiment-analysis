# credit to https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb

from GRU_Model import GRUModel
from get_embeddings import get_embeddings

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F

import math

from sklearn.model_selection import train_test_split

batch_size = 10

'''
    LOAD DATA
    '''
embeddings = get_embeddings()

train, test = train_test_split(embeddings[0], embeddings[1], test_size=0.2)

train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)

'''
    INSTANTIATE MODEL CLASS
    '''
input_dim = 1024
hidden_dim = 128
layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 10

# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

#######################
#  USE GPU FOR MODEL  #
#######################

if torch.cuda.is_available():
    model.cuda()

'''
    INSTANTIATE LOSS CLASS
    '''
criterion = nn.CrossEntropyLoss()

'''
    INSTANTIATE OPTIMIZER CLASS
    '''
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
    TRAIN THE MODEL
    '''

# Number of steps to unroll
seq_dim = 28

loss_list = []
iter = 0
for epoch in range(num_epochs):
    for i, (reviews, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            reviews = Variable(reviews.view(-1, seq_dim, input_dim).cuda())
            reviews = Variable(labels.cuda())
        else:
            reviews = Variable(reviews.view(-1, seq_dim, input_dim))
            labels = Variable(labels)
    
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(reviews)
        
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        if torch.cuda.is_available():
            loss.cuda()
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()
        
        loss_list.append(loss.item())
        iter += 1
        
        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for reviews, labels in test_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    reviews = Variable(reviews.view(-1, seq_dim, input_dim).cuda())
                else:
                    reviews = Variable(reviews.view(-1 , seq_dim, input_dim))
            
                # Forward pass only to get logits/output
                outputs = model(reviews)
                
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels
                total += labels.size(0)
                
                # Total correct predictions
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
        
            accuracy = 100 * correct / total

# Print Loss
print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
