#Indexing: 
# #Get a value at an index
import numpy as np
arr = np.arange(16)
print("arr : ", arr)

#Get a value at an index
print("Element at index 10 : ", arr[10])

#Get values in a range
print(arr[1:7])

#Slicing: 
# #Single dimension array
import numpy as np #IF RUNNING WHOLE REMOVE THIS IMPORT LIBRARY
a = np.arange(10)
b = a[2:7:2]
print(b)

# Multi dimensional array
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
print(a[1:])
print(a[1][2])
print(a[2])