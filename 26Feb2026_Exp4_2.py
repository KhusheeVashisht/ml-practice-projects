'''
26 feb 2026
Thursday(MA'am code)

K-Means Clustering Algorithm
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data= pd.read_csv('Iris.csv')

print(data.head())

X = data[['sepal_length','sepal_width']]
print(X)

kMeans = KMeans(n_clusters=3,random_state=0)

print("here it is ", kMeans)
kMeans.fit(X)

data['Cluster']=kMeans.labels_
print("Division of Cluster:" , data['Cluster'])

plt.scatter(X['sepal_length'],
            X['sepal_width'],
            c=data['Cluster'],
            cmap='viridis')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('K-Means Clustering on Iris Dataset')
plt.grid(True)
plt.show()