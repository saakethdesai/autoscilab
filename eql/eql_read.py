import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset

import sympy as sp

from sys import path

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


#Load pruned model
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
EPOCHS = 10001
lambda_reg = 1e-3

eql_model.load_state_dict(torch.load(f"eql_model_after_epoch_{EPOCHS-1}.pth", map_location=torch.device('cpu')))



input_size = X_train.shape[1]
x_symbols = sp.Matrix(sp.symbols(f'x0:{input_size}'))

# Extract weights from MyLayerFunc
W1 = eql_model.mylayer1.weight.detach().numpy()
act_list1 = eql_model.mylayer1.activation_list
layer1_out = W1 @ x_symbols


# Apply activations for MyLayerFunc
for k, act in enumerate(act_list1):
    if act == 'linear':
        layer1_out[k] = layer1_out[k]
    elif act == 'sin':
        layer1_out[k] = sp.sin(layer1_out[k])
    elif act == 'cos':
        layer1_out[k] = sp.cos(layer1_out[k])
    elif act == 'prod':
        layer1_out[k] = sp.prod(x_symbols)  # Product of all input variables


# Extract weights from MyLayerPoly
W2 = eql_model.mylayer2.weight.detach().numpy()
act_list2 = eql_model.mylayer2.activation_list
layer2_out = W2 @ layer1_out

# Apply activations for MyLayerPoly
for k, act in enumerate(act_list2):
    if act == 'linear':
        layer2_out[k] = layer2_out[k]
    elif act == 'power2':
        layer2_out[k] = layer2_out[k] ** 2
    elif act == 'power3':
        layer2_out[k] = layer2_out[k] ** 3
    elif act == 'sqrt':
        layer2_out[k] = sp.sqrt(layer2_out[k])
    elif act == 'prod':
        layer2_out[k] = sp.prod(layer1_out)  # Product of all activations

# Final output layer
W_out = eql_model.output_layer.weight.detach().numpy()
b_out = eql_model.output_layer.bias.detach().numpy()

# Compute final equation
output_eq = W_out @ layer2_out + b_out

# Pretty-print the equation
sp.pprint(output_eq, use_unicode=True)
