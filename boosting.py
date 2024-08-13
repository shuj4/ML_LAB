import numpy as np
import pandas as pd
import xgboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# Load the dataset
df = pd.read_csv("mushrooms.csv")
# Check column names
print("Columns in the DataFrame:", df.columns)
# Drop the 'veil-type' column if it exists
if "veil_type" in df.columns:
 df = df.drop("veil_type", axis=1)
else:
 print("'veil_type' column not found in the DataFrame")
# Display the first 6 rows of the DataFrame
df.head(6)
# Encode categorical features
label_encoder = LabelEncoder()
for column in df.columns:
 df[column] = label_encoder.fit_transform(df[column])
# Define features and target variable
X = df.loc[:, df.columns != 'type'] # Assuming 'type' is the target variable
Y = df['type'] # Set the target variable
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the AdaBoost model
adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=0.2).fit(X_train, Y_train)
# Evaluate the model
score = adaboost.score(X_test, Y_test)
print("Model accuracy:", score)