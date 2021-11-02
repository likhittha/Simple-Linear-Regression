# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting simple linear regression model to training set
from sklearn.linear_model import  LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting test set results
y_prediction = regressor.predict(x_test)

#visualising results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary based on experience (training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary based on experience (test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()