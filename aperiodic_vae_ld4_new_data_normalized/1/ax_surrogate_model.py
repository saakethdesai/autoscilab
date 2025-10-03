from ax import * 
from ax.modelbridge.registry import Models

from scipy.signal import savgol_filter 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from botorch.acquisition.monte_carlo import qExpectedImprovement

#https://ax.dev/tutorials/gpei_hartmann_developer.html

torch.manual_seed(0)

latent_dim = 4 

search_space = SearchSpace(
parameters = [ RangeParameter(name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=-3.0, upper=3.0) for i in range(latent_dim) ] )

class neural_net(nn.Module):
    def __init__(self):
        super(neural_net, self).__init__()
        self.ninputs = 4
        self.noutputs = 1
        self.linear1 = nn.Linear(self.ninputs, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 100)
        self.linear5 = nn.Linear(100, 100)
        self.linear6 = nn.Linear(100, 100)
        self.linear7 = nn.Linear(100, self.noutputs)
    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x) + x
        x = self.linear2(x)
        x = F.leaky_relu(x) + x
        x = self.linear3(x)
        x = F.leaky_relu(x) + x
        x = self.linear4(x)
        x = F.leaky_relu(x) + x
        x = self.linear5(x)
        x = F.leaky_relu(x) + x
        x = self.linear6(x)
        x = F.leaky_relu(x) + x
        x = self.linear7(x)
        x = F.relu(x)
        return x


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


class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        pos_list1 = ["-5.0", "-4.0", "-3.0", "-2.0", "-1.0", "0.0"]
        pos_list2 = ["1.0", "2.0", "3.0", "4.0"]
        pos_list = pos_list1 + pos_list2
        model_list = []
        for pos in pos_list:
            model_name = "../nn_" + pos + ".pth"
            model_list.append(model_name)
        for arm_name, arm in trial.arms_by_name.items():
            parameters = arm.parameters
            x = np.array([parameters.get(f"x{i}") for i in range(latent_dim)], dtype='float32')
            z = x.reshape((1, -1))
            z = torch.from_numpy(z)
            output_list = []
            for model_name in model_list:
                net = neural_net()
                net.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
                pump_profile = vae.decode(z)
                pump_profile = pump_profile.detach().numpy()
                pump_profile = savgol_filter(pump_profile, 81, 5)
                pump_profile = (pump_profile % (2*np.pi)) / (2*np.pi)
                output = net(z)
                output_np = output.detach().numpy()[0]
                output_list.append(output_np)
            #evaluate point
            obj = (output_list[-4]/sum(output_list))/np.mean(pump_profile) 
            trial_metadata["exp_result"] = obj
            xx = []
            for i in range(len(output_list)):
                xx.append(output_list[i][0])
            print (parameters, obj, xx)
        return trial_metadata


class MyMetric(Metric):
    def fetch_trial_data(self, trial): 
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                #"mean": (params["x1"] + 2*params["x2"] - 7)**2 + (2*params["x1"] + params["x2"] - 5)**2,
                "mean": trial.run_metadata["exp_result"],
                "sem": 0.0
                })
        return Data(df=pd.DataFrame.from_records(records)) 


vae = VAE()
vae.load_state_dict(torch.load("../apvaeld4.pth",map_location=torch.device('cpu')))


#param_names = [f"x{i}" for i in range(latent_dim)]
optimization_config = OptimizationConfig(objective = Objective(metric=MyMetric(name="mymetric"), minimize=False))
exp = Experiment(name="test", search_space=search_space, optimization_config=optimization_config, runner=MyRunner())

NUM_SOBOL_TRIALS = 1000 
NUM_BOTORCH_TRIALS = 500 

#print(f"Running Sobol initialization trials...")
sobol = Models.SOBOL(search_space=exp.search_space)

for i in range(NUM_SOBOL_TRIALS):
    print(f"Running SOBOL trial {i + 1}/{NUM_SOBOL_TRIALS}...")
    # Produce a GeneratorRun from the model, which contains proposed arm(s) and other metadata
    generator_run = sobol.gen(n=1) #?parallelization
    # Add generator run to a trial to make it part of the experiment and evaluate arm(s) in it
    trial = exp.new_trial(generator_run=generator_run)
    # Start trial run to evaluate arm(s) in the trial
    trial.run()
    # Mark trial as completed to record when a trial run is completed 
    # and enable fetching of data for metrics on the experiment 
    # (by default, trials must be completed before metrics can fetch their data,
    # unless a metric is explicitly configured otherwise)
    trial.mark_completed()

for i in range(NUM_BOTORCH_TRIALS):
    print(f"Running GP+EI optimization trial {i + 1}/{NUM_BOTORCH_TRIALS}...")
    # Reinitialize GP+EI model at each step with updated data.
    gpei = Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data(), botorch_acqf_class=qExpectedImprovement)
    generator_run = gpei.gen(n=1)
    trial = exp.new_trial(generator_run=generator_run)
    trial.run()
    trial.mark_completed()

print("Done!")

