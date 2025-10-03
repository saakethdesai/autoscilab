import sympy as sp
import numpy as np
from gplearn.genetic import SymbolicRegressor

# Load data from the file
data = np.loadtxt("d_vdp1.txt", delimiter=',', skiprows=1)

# Extract columns
labels = data[:, 0:1]
x_vals = data[:, 1:2]
y_vals = data[:, 2:3]

X = np.hstack((x_vals, y_vals))
y = labels
print (X.shape, y.shape)

est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

est_gp.fit(X, y)

print(est_gp._program)