import pandas as pd
df = pd.read_csv("diabetes.csv")
df.head()
df.isnull().sum()
X = df.drop("Outcome",axis="columns")
y = df.Outcome
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:3]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)
X_train.shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RandomForestClassifier(n_estimators=50), X, y, cv=5)
scores.mean()