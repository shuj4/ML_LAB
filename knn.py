from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy : ", acc)
new_instance = [[5.0, 3.5, 1.4, 2.2]]
predicted = knn.predict(new_instance)
print("Predicted class ", predicted)