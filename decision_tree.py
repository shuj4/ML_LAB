!pip install graphviz
!apt install libgraphviz-dev
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = np.array([
['famous engg works', 11155,155000],
['asset flip', 11134,33100],
['bmw', 555,10340],
['mjcet', 1332,10400],
['tax', 1141,12000],
['jeee', 1411,10040],
['yoyo', 111,10200],
[' flip', 1141,1000],
['nessain', 1161,10400],
['asset', 1611,10040],
])
print(dataset)
x = dataset[:,1:2].astype(int)
y = dataset[:,2].astype(int)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)
y_pred = regressor.predict([[111]])
print (y_pred)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color ='red')
plt.plot(x_grid,regressor.predict(x_grid), color ='blue')
plt.title('business analytics')
plt.xlabel('cost')
plt.ylabel('profit')
plt.show()
from sklearn.tree import export_graphviz
export_graphviz(regressor ,out_file ='tree.dot',feature_names=['production cost'])