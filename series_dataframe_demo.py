import pandas as pd
fruits = ['apples', 'oranges', 'cherries', 'pears']
quantities = [20, 33, 52, 10]
S = pd.Series(quantities, index=fruits)
print(S)
# Creating a DataFrame from a dictionary
data = {
 'Name': ['Alice', 'Bob', 'Charlie', 'David'],
 'Age': [25, 30, 22, 28],
 'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
print(df)