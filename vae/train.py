import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from glob import glob

from datagen import *
import os

torch.manual_seed(0)
np.random.seed(0)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.image_width = 3840
        self.latent_dims = 4 
        self.e1 = nn.Linear(self.image_width,int(self.image_width/2))
        self.e2 = nn.Linear(int(self.image_width/2),int(self.image_width/4))
        self.e3 = nn.Linear(int(self.image_width/4),int(self.image_width/8))
        self.e4 = nn.Linear(int(self.image_width/8),int(self.image_width/16))
        self.e5 = nn.Linear(int(self.image_width/16),int(self.image_width/32))       
        self.e6 = nn.Linear(int(self.image_width/32),int(self.image_width/64))
        self.e7 = nn.Linear(int(self.image_width/64),int(self.image_width/128))
        self.e8 = nn.Linear(int(self.image_width/128),int(self.image_width/256))
        self.elinear1 = nn.Linear(int(self.image_width/256),int(self.latent_dims))
        self.elinear2 = nn.Linear(int(self.image_width/256),int(self.latent_dims))
                                                                                                
        ## decoder Layers
        
        self.d1 = nn.Linear(int(self.image_width/2),int(self.image_width))
        self.d2 = nn.Linear(int(self.image_width/4),int(self.image_width/2))
        self.d3 = nn.Linear(int(self.image_width/8),int(self.image_width/4))
        self.d4 = nn.Linear(int(self.image_width/16),int(self.image_width/8))
        self.d5 = nn.Linear(int(self.image_width/32),int(self.image_width/16))       
        self.d6 = nn.Linear(int(self.image_width/64),int(self.image_width/32))
        self.d7 = nn.Linear(int(self.image_width/128),int(self.image_width/64))
        self.d8 = nn.Linear(int(self.image_width/256),int(self.image_width/128))
        self.dlinear1 = nn.Linear(int(self.latent_dims),int(self.image_width/256))
        
    def encode(self, x):
        x = self.e1(x)
        x = F.leaky_relu(x)
        x = self.e2(x)
        x = F.leaky_relu(x)
        x = self.e3(x)
        x = F.leaky_relu(x)
        x = self.e4(x)
        x = F.leaky_relu(x)
        x = self.e5(x)
        x = F.leaky_relu(x)
        x = self.e6(x)
        x = F.leaky_relu(x)
        x = self.e7(x)
        x = F.leaky_relu(x)
        x = self.e8(x)
        x = F.leaky_relu(x)
        mean = self.elinear1(x)
        logvar = self.elinear2(x)
        return mean, logvar

    def decode(self, z): 
        x = self.dlinear1(z)
        x = F.leaky_relu(x)
        x = self.d8(x)
        x = F.leaky_relu(x)
        x = self.d7(x)
        x = F.leaky_relu(x)
        x = self.d6(x)
        x = F.leaky_relu(x)
        x = self.d5(x)
        x = F.leaky_relu(x)
        x = self.d4(x)
        x = F.leaky_relu(x)
        x = self.d3(x)
        x = F.leaky_relu(x)
        x = self.d2(x)
        x = F.leaky_relu(x)
        x = self.d1(x)
        x = torch.relu(x)
        return x 
	
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar 


def vae_loss(recon_x, x, mu, logvar):
    mse = F.mse_loss(recon_x, x)
    kld = torch.mean(-0.5*(torch.sum(1 + logvar - mu**2 - torch.exp(logvar), axis=1)), axis=0)
    derivative = np.mean(np.abs(np.gradient(recon_x.detach().numpy())[0]))
    derivative_torch = torch.as_tensor(derivative)
    elbo = mse + kld #+ derivative_torch
    print ("LOSS = ", mse.detach().numpy(), kld.detach().numpy(), derivative_torch.detach().numpy())
    return elbo 


#----------------------------------------------#
#create model
latent_dim = 4 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device=device)
num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)


learning_rate = 1e-4
optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate)

EPOCHS = 1000

print('Training ...')

cwd = os.getcwd()

for epoch in range(EPOCHS):

    train_loss = 0
    nbatches_train = 90 
    
    val_loss = 0
    nbatches_val = 5 
    
    test_loss = 0
    nbatches_test = 5 

    for batch_idx in range(nbatches_train):
        
        train_data = generate_raw_batch_data(N=64, seed=batch_idx)
        train_data = train_data.astype('float32')
        #print (train_data.shape)
        train_data = torch.from_numpy(train_data)
        train_data = train_data.to(device)
        
        optimizer.zero_grad()
        # vae reconstruction
        train_data_recon, mu, logvar = vae(train_data)
        # reconstruction error
        loss = vae_loss(train_data_recon, train_data, mu, logvar)
        # backpropagation
        loss.backward()
        curr_loss = loss.item()
        train_loss += curr_loss
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        print('Batch [%d / %d] ELBO loss: %f' % (batch_idx+1, nbatches_train, curr_loss))

        
        
    for batch_idx in range(nbatches_val):
        val_data = generate_raw_batch_data(N=64, seed=batch_idx+90)
        val_data = val_data.astype('float32')
        #print (batch_idx+90, val_data.shape)
        val_data = torch.from_numpy(val_data)
        val_data = val_data.to(device)
        # vae reconstruction
        val_data_recon, mu, logvar = vae(val_data)
        # reconstruction error
        loss = vae_loss(val_data_recon, val_data, mu, logvar)
        curr_loss = loss.item()
        val_loss += curr_loss
        
    for batch_idx in range(nbatches_test):
        test_data = generate_raw_batch_data(N=64, seed=batch_idx+95)
        test_data = test_data.astype('float32')
        #print (batch_idx+95, test_data.shape)
        test_data = torch.from_numpy(test_data)
        test_data = test_data.to(device)
        # vae reconstruction
        test_data_recon, mu, logvar = vae(test_data)
        # reconstruction error
        loss = vae_loss(val_data_recon, test_data, mu, logvar)
        curr_loss = loss.item()
        test_loss += curr_loss
    
    train_loss /= nbatches_train 
    val_loss /= nbatches_val
    test_loss /= nbatches_test
    print('Epoch [%d / %d] ELBO loss train, val, test: %f %f %f' % (epoch+1, EPOCHS, train_loss, val_loss, test_loss))

    torch.save(vae.state_dict(), "vae.pth")
