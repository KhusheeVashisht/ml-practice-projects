'''
Docstring for 5Feb2026
implement Non-Linear regression with and without dataset
'''
#Polynomial Model (without Dataset)
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])
y = np.array([2,5,10,17,26])

coeffs = np.polyfit(x, y, 2)
model = np.poly1d(coeffs)

plt.scatter(x, y)
plt.plot(x, model(x))
plt.show()

#Exponential Model (Without Database)
from scipy.optimize import curve_fit

def model(x, a, b):
    return a * np.exp(b * x)

params, _ = curve_fit(model, x, y)
