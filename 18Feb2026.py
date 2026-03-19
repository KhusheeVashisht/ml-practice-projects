'''
Docstring for 18Feb2026
exp 3
Execute an SVM model and evaluate its performance using Python:
a) Split the dataset into training and testing sets.
b) Apply SVM with any kernel (linear/RBF/poly).
c) Print the accuracy score and classification report.

'''
'''
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#load the iris dataset

iris = datasets.load_iris()
X=iris.data #features
y= iris.target #labels

#split the dataset into training and testing sets
X_train , X_test, y_train , y_test = train_test_split(X,y ,test_size=0.2, random_state=48)

#create an svm model with linear kernle
svm_model= SVC(kernel='linear',C=1.0,random_state=48)

#train the model 
svm_model.fit(X_train,y_train)

#make predictions
y_pred = svm_model.predict(X_test)

#evaluate the model
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy:.2f}")
print("Classification_report:")
print(classification_report(y_test,y_pred))
'''
'''

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#load the iris dataset

iris = datasets.load_iris()
X=iris.data #features
y= iris.target #labels

#split the dataset into training and testing sets
X_train , X_test, y_train , y_test = train_test_split(X,y ,test_size=0.2, random_state=48)

#create an svm model with linear kernle
svm_model= SVC(kernel='rbf',C=1.0,random_state=48)

#train the model 
svm_model.fit(X_train,y_train)

#make predictions
y_pred = svm_model.predict(X_test)

#evaluate the model
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy:.2f}")
print("Classification_report:")
print(classification_report(y_test,y_pred))
'''
# for 2d plotting 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#load the iris dataset

iris = datasets.load_iris()
X=iris.data[:,:2] #Select any two features : sepal lenght and sepal width
y= iris.target #labels

#split the dataset into training and testing sets
X_train , X_test, y_train , y_test = train_test_split(X,y ,test_size=0.2, random_state=48)

#Standarize features
scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


#create an svm model with linear kernle
svm_model= SVC(kernel='linear',C=1.0,random_state=48)#rbf = radial basis function

#train the model 
svm_model.fit(X_train,y_train)

#create a mesh to plot decision boundaries
x_min,x_max = X_train[:,0].min() -1,X_train[:,0].max() +1
y_min,y_max = X_train[:,0].min() -1,X_train[:,0].max() +1
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min, y_max,0.01))

#predict for mesh points
Z=svm_model.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)

#Plot decision boundaries
plt.contourf(xx,yy,Z,alpha=0.3,cmap=plt.cm.coolwarm)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.coolwarm,marker="o",label="Train Data")
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=plt.cm.coolwarm,marker="x",label="Test Data")

plt.xlabel("Sepal Length (Standarized)")
plt.ylabel("Sepal width (Standarized)")
plt.title("SVM Decision Boundary on Iris Dataset (2 features)")
plt.legend()
plt.show()
#make predictions
y_pred = svm_model.predict(X_test)

#evaluate the model
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy:.2f}")
print("Classification_report:")
print(classification_report(y_test,y_pred))