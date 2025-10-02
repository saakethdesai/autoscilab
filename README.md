# Autoscilab
Code for the machine learning components of AutoSciLab

This directory contains the scripts used in this work:
The directory structure is as follows:
vae/ - contains training and evaluation scripts for the VAE
poly_slm/ - contains experimental results sweeping over grating and curvature for different angles
active_learning/ - contains multiple iterations of the active learning loop, over multiple data 
nn_surrogate_model/ - contains scripts training the neural network surrogate model used to predict directivity when running the active learning loop. Note that this is done as an alternate approach to directly running the experiment
eql/ - contains notebooks and scripts that document training of the equation learner network. It also contains scripts to check equation learning for some other popular packages such as PySR.
