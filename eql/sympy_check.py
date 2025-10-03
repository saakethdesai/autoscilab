import sympy as sp
import numpy as np

# Load data from the file
data = np.loadtxt("d_vdp1.txt", delimiter=',', skiprows=1)

# Extract columns
labels = data[:, 0]
x_vals = data[:, 1]
y_vals = data[:, 2]

# Define symbolic variables
x, y, a, b, c, d = sp.symbols('x y a b c d')

# Define the equation to fit
equation = a * x + b * y + c + d * x**3

# Define the error function
error = 0
for i in range(len(labels)):
    subs_eqn = equation.subs({x: x_vals[i], y: y_vals[i]})
    error += (labels[i] - subs_eqn)**2 
error /= len(labels)

# Solve for the coefficients that minimize the error
solution = sp.solve(sp.derive_by_array(error, [a, b, c, d]), [a, b, c, d])

# Print the fitted equation
fitted_equation = equation.subs(solution)
print("Fitted Equation:", fitted_equation)