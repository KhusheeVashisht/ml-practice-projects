'''
Implementing y= 1/1+e^-x
'''
import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate x values
x = np.linspace(-10, 10, 100)

# Compute sigmoid values
y = sigmoid(x)

# Plot the sigmoid curve
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.title("Sigmoid Function Curve")
plt.show()
