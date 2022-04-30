#Decision Tree Regression
# Import the necessary modules and libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
#Import papa johns data
dataset = pd.read_csv('pjData.csv')
dataset.fillna(0, inplace = True) #clean data so that there is no misinput {as in there is no empty values}
#Obtain input and output 
#MODULARITY CHANGES HERE
X = dataset.iloc[:, 18:21].values   #target features 18:21 {From the pjData.csv}
y = dataset.iloc[:, 22].values      #The testing parameter {You can change this value to change the testing parameter}
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
# Predict
X_t = np.arange(0.0, 15.0, 0.01)[:, np.newaxis]
X_test = X_t.reshape([500,3])
y_1 = regr_1.predict(X_test) #this will be the regression value of the prediction line
# Plot the results
plt.figure()
plt.scatter(X[:,0], y, s=20, edgecolor="blue", c="blue", label="f1")
plt.scatter(X[:,1], y, s=20, edgecolor="cyan", c="cyan", label="f2")
plt.scatter(X[:,2], y, s=20, edgecolor="purple", c="purple", label="f3")
plt.plot(X_test, y_1, color="red", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
#analysis, close the scatter graph in order to print out the accuracy
vgh = regr_1.score(X,y)
print(("Accuracy is " + str(vgh)))