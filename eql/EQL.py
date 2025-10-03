import torch
import torch.nn as nn

import numpy as np


class MyLayerFunc(nn.Module):
    
    def __init__(self, size_in, size_out, activation_list):
        super().__init__()
        self.size_in, self.size_out, self.activation_list = size_in, size_out, activation_list
        weight = torch.ones(self.size_out, self.size_in)
        self.weight = nn.Parameter(weight)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nrep = int(size_out/len(activation_list))
        self.nact = len(self.activation_list)
        self.activation_list = self.activation_list * nrep
        
    def activation(self, act, x):
        if (act == 'linear'):
            answer = x
        if (act == 'cos'):
            answer = torch.cos(x)
        if (act == 'sin'):
            answer = torch.sin(x)
        if (act == 'tan'):
            answer = torch.tan(x)
        return answer
    
    def forward(self, x):
        
        w_times_x = torch.mm(x, self.weight.t())
        linear_combo = w_times_x
        
        output = torch.zeros(linear_combo.shape)

        for k in range(self.nact):
            act = self.activation_list[k]
            idx = [i for i in range(k, self.size_out, self.nact)]
            if (act == 'prod'):
                input_product = torch.prod(x[:, ], dim=1, keepdim=True) #Nx1
                weight_product = torch.prod(self.weight[idx, :], dim=1, keepdim=True).T #repeatx1
                output[:, idx] = torch.outer(input_product.flatten(), weight_product.flatten())
            else:
                output[:, idx] = self.activation(act, linear_combo[:, idx])
            
        #contribution calculation
        #averge neuron values over a batch
        average_neuron_values = torch.mean(x, 0, keepdim=False)
        #create a diagonal neuron value matrix
        average_neuron_values_diag = torch.diag(average_neuron_values)
        #contribution = W * x
        self.contribution = nn.Parameter(torch.mm(self.weight, average_neuron_values_diag))
        
        return output

    
class MyLayerPoly(nn.Module):
    
    def __init__(self, size_in, size_out, activation_list):
        super().__init__()
        self.size_in, self.size_out, self.activation_list = size_in, size_out, activation_list
        weight = torch.ones(self.size_out, self.size_in)
        self.weight = nn.Parameter(weight)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nrep = int(size_out/len(activation_list))
        self.nact = len(self.activation_list)
        self.activation_list = self.activation_list * nrep
        
    def activation(self, act, x):
        if (act == 'linear'):
            answer = x
        if (act == 'power2'):
            answer = x**2
        if (act == 'power3'):
            answer = x**3
        if (act == 'sqrt'):
            answer = x**0.5
        return answer
    
    def forward(self, x):
        
        w_times_x = torch.mm(x, self.weight.t())
        linear_combo = w_times_x
        
        output = torch.zeros(linear_combo.shape)
        
        for k in range(self.nact):
            act = self.activation_list[k]
            idx = [i for i in range(k, self.size_out, self.nact)]
            if (act == 'prod'):
                input_product = torch.prod(x[:, ], dim=1, keepdim=True) #Nx1
                weight_product = torch.prod(self.weight[idx, :], dim=1, keepdim=True).T #repeatx1
                output[:, idx] = torch.outer(input_product.flatten(), weight_product.flatten())
            else:
                output[:, idx] = self.activation(act, linear_combo[:, idx])
        
        #contribution calculation
        #averge neuron values over a batch
        average_neuron_values = torch.mean(x, 0, keepdim=False)
        #create a diagonal neuron value matrix
        average_neuron_values_diag = torch.diag(average_neuron_values)
        #contribution = W * x
        self.contribution = nn.Parameter(torch.mm(self.weight, average_neuron_values_diag))

        return output