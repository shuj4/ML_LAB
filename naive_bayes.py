import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 50)
naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)
y_pred = naive_bayes.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy : ", acc)
new_instance = [[5.0, 3.5, 1.4, 2.2]]
predicted = naive_bayes.predict(new_instance)
print("Predicted class ", predicted)