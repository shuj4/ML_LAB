from scipy import linalg
import numpy as np
#Making the numpy array
mat = np.array([[1,2,3],[3,4,5],[7,8,6]])
#Passing the matrix to the det function
mat1 = linalg.det(mat)
#printing the determinant
print(f'Determinant of the matrix\n{mat} \n is {mat1}')
#finding the inverse of the matrix using the inv() function
mat2 = linalg.inv(mat)
#printing the original and inverse matrices
print(f'Inverse of the matrix\n{mat} \n is \n{mat2}')