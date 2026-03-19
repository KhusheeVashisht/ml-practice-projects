# Experiment 1: Implementation of Python Basic Libraries

# ------------------ Math Library ------------------
import math

print("---- Math Library ----")
num = 25
print("Square root:", math.sqrt(num))
print("Power:", math.pow(num, 2))
print("Factorial:", math.factorial(5))


# ------------------ NumPy Library ------------------
import numpy as np

print("\n---- NumPy Library ----")
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print("Array a:", a)
print("Array b:", b)
print("Addition:", a + b)
print("Mean of a:", np.mean(a))
print("Standard Deviation of a:", np.std(a))


# ------------------ Matplotlib Library ------------------
import matplotlib.pyplot as plt

print("\n---- Matplotlib Library ----")
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

plt.figure()
plt.plot(x, y)
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Simple Line Plot using Matplotlib")
plt.show()


# ------------------ Seaborn Library ------------------
import seaborn as sns

print("\n---- Seaborn Library ----")
flights = sns.load_dataset("flights")
print(flights.head())

sns.relplot(data=flights, x="month", y="passengers", kind="line")
plt.show()


# ------------------ SciPy Library ------------------
from scipy import stats

print("\n---- SciPy Library ----")
data = [12, 15, 20, 22, 25, 30]

print("Mean:", stats.tmean(data))
print("Standard Deviation:", stats.tstd(data))
print("Variance:", stats.tvar(data))
