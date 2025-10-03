import numpy as np
from subprocess import call, check_output
import os

from matplotlib import pyplot as plt


def generate_raw_batch_data(N, seed):
    batch_data = []
    np.random.seed(seed)
    a = np.random.uniform(-800, 800, N).reshape(-1, 1)
    b = np.random.uniform(-800, 800, N).reshape(-1, 1)
    c = np.random.uniform(-800, 800, N).reshape(-1, 1)

    x = np.linspace(0, 1, 3840)
    x = np.repeat(x.reshape(1, len(x)), N, axis=0)
    x0 = np.random.uniform(0, 1, N).reshape(-1, 1)
    
    data = a*(x-x0)**2 + b*(x-x0) + c*(np.abs(x-x0))**0.5
    min_value = np.min(data, axis=1).reshape(-1, 1) 
    min_array = np.repeat(min_value, 3840, axis=1)
    data = data - min_array #+ 1
    #log_data = np.log10(data)
    return data 


