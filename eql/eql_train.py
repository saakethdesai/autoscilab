import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset

from sympy import *

from sys import path
#path.append('/Users/saadesa/Research/PNN FCC/parsimonious-nn/EQL_connection_pruning')

from EQL import *


np.random.seed(0)
torch.manual_seed(0)


# Load data from the file
data = np.loadtxt("d_vdp1.txt", delimiter=',', skiprows=1)

# Extract columns
labels = data[:, 0:1]
x_vals = data[:, 1:2]
y_vals = data[:, 2:3]

X = np.hstack((x_vals, y_vals))
y = labels
print (X.shape, y.shape)

# Split data into training and validation sets
split_index = int(0.9 * len(X))
X_train = X[:split_index]
Y_train = y[:split_index]
X_val = X[split_index:]
Y_val = y[split_index:]

# Convert to PyTorch tensors
X_train_T = torch.from_numpy(X_train).float()
Y_train_T = torch.from_numpy(Y_train).float()
X_val_T = torch.from_numpy(X_val).float()
Y_val_T = torch.from_numpy(Y_val).float()

# Create TensorDatasets
training_dataset = TensorDataset(X_train_T, Y_train_T)
validation_dataset = TensorDataset(X_val_T, Y_val_T)

# Create DataLoaders
train_data = DataLoader(training_dataset, batch_size=len(X_train), shuffle=False)
val_data = DataLoader(validation_dataset, batch_size=len(X_val), shuffle=False)


#Create EQLNet
class EQLNet(nn.Module):

    def __init__(self, act, n_rep=2):
        super().__init__()

        #activation_list_func = ['linear', 'sin', 'cos', 'prod']
        activation_list_func = ['linear', 'linear', 'linear', 'linear']
        #act = 0 (all linear); 1 = (prod); 2 = (x**2)
        if (act == 0):
            activation_list_poly = ['linear', 'linear', 'linear', 'linear']
        elif (act == 1):
            activation_list_poly = ['linear', 'linear', 'linear', 'prod']
        elif (act == 2):
            activation_list_poly = ['linear', 'linear', 'power3', 'linear']
        elif (act == 3):
            activation_list_poly = ['linear', 'linear', 'power2', 'prod']

        self.mylayer1 = MyLayerFunc(X_train.shape[1], len(activation_list_func)*n_rep, activation_list_func)        
        self.mylayer2 = MyLayerPoly(len(activation_list_func)*n_rep, 
                                    len(activation_list_poly)*n_rep, activation_list_poly)
        self.output_layer = nn.Linear(len(activation_list_poly)*n_rep, 1)

    def forward(self, x):

        x = self.mylayer1.forward(x)
        x = self.mylayer2.forward(x)
        x = self.output_layer(x)

        return x
    
def loss_function(predict_y, y):
    loss = F.mse_loss(predict_y, y)
    return loss


#Create unpruned model
print ("Creating model")

eql_model = EQLNet(act=2, n_rep=2)
#pass dummy inputs to initialize contributions parameter
dummy_inputs = torch.zeros((1, 2))
dummy_outputs = eql_model(dummy_inputs)

learning_rate = 1e-3
optimizer = torch.optim.Adam(params=eql_model.parameters(),lr = learning_rate)
num_params = sum(p.numel() for p in eql_model.parameters())
print(num_params)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5001
lambda_reg = 1e-3

#Train unpruned model
print ("Train unpruned model")

train_losses = []
val_losses = []

for epoch in range(EPOCHS):

    for x, y in train_data:
        optimizer.zero_grad()    
        predict_y = eql_model(x)

        #regularization = compute_regularization(eql_model.named_parameters())

        total_loss_train = loss_function(predict_y, y) #+ lambda_reg*regularization
        total_loss_train.backward()
        optimizer.step()

    for x, y in val_data:
        optimizer.zero_grad()    
        predict_y = eql_model(x)

        #regularization = compute_regularization(eql_model.named_parameters())

        total_loss_val = loss_function(predict_y, y) #+ lambda_reg*regularization
        #total_loss_val.backward()
        #optimizer.step()

    if (epoch % 100 == 0):
        print(f'epoch = {epoch} - train loss = {total_loss_train} - val loss = {total_loss_val}')

    train_losses.append(total_loss_train.detach().numpy())
    val_losses.append(total_loss_val.detach().numpy())

    if epoch % (EPOCHS-1) == 0:
        torch.save(eql_model.state_dict(), f'nn_model_after_epoch_{epoch}.pth')


#Define pruning scheme 1

class MyPruningMethod(prune.BasePruningMethod):
    
    PRUNING_TYPE = 'unstructured'
    
    def __init__(self, amount, current_mask):
        self.amount = amount
        self.current_mask = current_mask
    
    def compute_mask(self, t, default_mask):
        
        #find indices that weren't zeroed before
        nonzero_indices = torch.nonzero(self.current_mask.view(-1)).flatten()
        
        #find topk values out of non-zero indices
        number = int(self.amount*len(t.view(-1)))
        topk = torch.topk(torch.abs(t).view(-1)[nonzero_indices], k=number, largest=False)
        
        #set topk indices to zero
        tmp_mask = self.current_mask.view(-1)[nonzero_indices]
        tmp_mask[topk.indices] = 0
        self.current_mask.view(-1)[nonzero_indices] = tmp_mask
        
        return self.current_mask
    
    def display_mask(self):
        
        return self.current_mask
    
def myprune(module, name, amount, current_mask):

    MyPruningMethod.apply(module, name, amount=amount, current_mask=current_mask)
    
    prune.remove(module, name)
    
    return module

#Define pruning scheme 2

class SetZeroPruningMethod(prune.BasePruningMethod):
    
    PRUNING_TYPE = 'unstructured'
    
    def __init__(self, amount, current_mask):
        self.amount = amount
        self.current_mask = current_mask
    
    def compute_mask(self, t, default_mask):
        
        #find indices that were zeroed before
        nonzero_indices = torch.nonzero(self.current_mask.view(-1)).flatten()
        nonzero_indices_list = nonzero_indices.tolist()
        total_index_list = [i for i in range(len(self.current_mask.view(-1)))]
        
        zero_indices_list = list(set(total_index_list) - set(nonzero_indices_list))
        zero_indices = torch.LongTensor(zero_indices_list)
        
        
        #set zero indices to zero
        self.current_mask.view(-1)[zero_indices] = 0
        
        return self.current_mask

def setzero(module, name, amount, current_mask):

    SetZeroPruningMethod.apply(module, name, amount=amount, current_mask=current_mask)
    
    prune.remove(module, name)
    
    return module


# def compute_regularization(params):
#     constant_list = [0, 1]
#     regularization = 0
#     for name, p in params:
#         if "contribution" not in name:
#             term = 1
#             #product over all constants
#             for constant in constant_list:
#                 term_current_constant = (p.view(-1) - constant)**2
#                 term = term * term_current_constant
#             #average over all weights
#             regularization += torch.sum(term)/len(term)
#     return regularization

#Reload model for pruning
print ("Load model")

print (eql_model.mylayer1.contribution)

eql_model = EQLNet(act=2, n_rep=2)

#pass dummy inputs to initialize contributions parameter
dummy_inputs = torch.zeros((1, 2))
dummy_outputs = eql_model(dummy_inputs)

eql_model.load_state_dict(torch.load("nn_model_after_epoch_5000.pth", map_location=torch.device('cpu')))

print (eql_model.mylayer1.contribution)

k = 0.1
moving_average_loss = 10
layer_list = list(eql_model.modules())[1:-1]
current_mask_list = [torch.ones_like(layer_list[i].weight) for i in range(len(layer_list))]
sparsity_list = [0 for i in range(len(layer_list))]
target_sparsity_list = [0.9, 0.9]

losses = []

learning_rate = 1e-3
optimizer = torch.optim.Adam(params=eql_model.parameters(),lr = learning_rate)
num_params = sum(p.numel() for p in eql_model.parameters())
print(num_params)

EPOCHS = 10001

#Train with pruning
print ("Train with pruning")

for epoch in range(EPOCHS):

    for x,y in train_data:
        #train
        optimizer.zero_grad()
        predict_y = eql_model(x)

        #regularization = compute_regularization(nn_model.named_parameters())

        total_loss = loss_function(predict_y,y) #+ lambda_reg*regularization
        total_loss.backward()
        losses.append(total_loss.detach().numpy())
        optimizer.step()

        #reset weights to zero
        for idx, layer in enumerate(layer_list):
            setzero(layer, 'weight', k, current_mask=current_mask_list[idx])

        if epoch > 10:
            moving_average_loss = np.mean(np.array(losses)[-5:])

        if moving_average_loss < 1e-2:
            #prune
            for idx, layer in enumerate(layer_list):
                if (sparsity_list[idx] < target_sparsity_list[idx]):
                    #print (f'pruning layer {layer}')
                    dummy_mypruningmethod_object = MyPruningMethod(k, current_mask_list[idx])
                    myprune(layer, 'contribution', k, current_mask=current_mask_list[idx])
                    current_mask_list[idx] = dummy_mypruningmethod_object.display_mask()
                    sparsity_list[idx] += k
                    sparsity_list[idx] = np.round(sparsity_list[idx], 1)
                    print (f'Current sparsity {sparsity_list[idx]}')
                
    if (epoch % 100 == 0):
        print(f'epoch = {epoch} - loss = {total_loss}')

    if epoch % (EPOCHS-1) == 0:
        torch.save(eql_model.state_dict(), f'eql_model_after_epoch_{epoch}.pth')


