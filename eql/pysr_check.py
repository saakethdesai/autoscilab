import sympy as sp
import numpy as np
from pysr import PySRRegressor

# Load data from the file
data = np.loadtxt("d_vdp1.txt", delimiter=',', skiprows=1)

# Extract columns
labels = data[:, 0:1]
x_vals = data[:, 1:2]
y_vals = data[:, 2:3]

X = np.hstack((x_vals, y_vals))
y = labels
print (X.shape, y.shape)

print ("Reached")

model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

#model.fit(X, y)