import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Extract sepal length (feature 0) and sepal width (feature 1)
X_sepal = X[:, [0, 1]] # Select sepal length and sepal width
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sepal, y, test_size=0.20, random_state=40)
# Create a LogisticRegression instance
logreg = LogisticRegression(max_iter=200)
# Fit the training data to the model
logreg.fit(X_train, y_train)
# Predict the labels for the test data
y_pred = logreg.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Plotting
plt.figure(figsize=(10, 6))
# Scatter plot of Sepal Length vs Sepal Width
plt.scatter(X_sepal[:, 0], X_sepal[:, 1], c=y, edgecolor='k', s=50, cmap=plt.cm.RdYlBu, marker='o',
label='Data')
# Set labels and title
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')
plt.legend()
plt.show()