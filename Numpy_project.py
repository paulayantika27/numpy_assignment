#Question 1:
import numpy as np

arr = np.arange(1, 26).reshape(5, 5)
print(arr)

#Question 2:
identity = np.eye(4)
print(identity)

#Question 3:
arr = np.arange(100, 201, 10)
print(arr)

#Question 4:
matrix = np.random.rand(3,3)
det = np.linalg.det(matrix)
print(matrix)
print("Determinant:", det)

#Question 5:
rand_ints = np.random.randint(1, 101, size=10)
print(rand_ints)
mean = np.mean(rand_ints)
median = np.median(rand_ints)
std_dev = np.std(rand_ints)
print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)

#Question 6:
arr = np.arange(1, 13).reshape(3, 4)
print(arr)

#Question 7:
A = np.random.randint(1, 10, (3,3))
B = np.random.randint(1, 10, (3,3))
C = np.dot(A, B)
print("A:\n", A, "\nB:\n", B, "\nA*B:\n", C)

#Question 8:
M = np.array([[4, 2], [1, 3]])
values, vectors = np.linalg.eig(M)
print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)

#Question 9:
mat = np.random.rand(5,5)
diag = np.diag(mat)
print(diag)

#Question 10:
arr = np.array([10, 20, 30, 40, 50])
norm = (arr - arr.min()) / (arr.max() - arr.min())
print(norm)

#Question 11:
arr = np.random.randint(1, 20, (3,3))
print("Original:\n", arr)
print("Row-wise sort:\n", np.sort(arr, axis=1))
print("Column-wise sort:\n", np.sort(arr, axis=0))

#Question 12:
arr = np.random.randint(1, 100, 10)
print("Array:", arr)
print("Max index:", np.argmax(arr))
print("Min index:", np.argmin(arr))
print("Sorted indices:", np.argsort(arr))

#Question 13:
arr = np.array([[1,2],[3,4]])
print("ravel:", arr.ravel())
print("flatten:", arr.flatten())
print("transpose:\n", arr.T)

#Question 14:
mat = np.random.randint(1, 10, (3,3))
inv = np.linalg.inv(mat)
print(inv)

#Question 15:
perm = np.random.permutation(np.arange(1, 11))
print(perm)

#Question 16:
arr = np.arange(21)
arr[arr%2==0] = -1
print(arr)

#Question 17:
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.dot(a,b))

#Question 18:
mat = np.random.randint(1,10,(5,5))
print(np.trace(mat))

#Question 19:
arr = np.arange(9)
parts = np.split(arr, 3)
print(parts)

#Question 20:
arr = np.random.rand(3,3,3)
print(arr.mean(axis=0))

#Question 21:
arr = np.array([1,2,3,4])
print(np.cumsum(arr))

#Question 22:
mat = np.random.randint(1,10,(4,4))
print(np.triu(mat))

#Question 23:
checker = np.indices((6,6)).sum(axis=0) % 2
print(checker)

#Question 24:
mat = np.random.rand(3,3)
print(np.sqrt(mat))

#Question 25:
arr = np.arange(20)
print(np.flip(arr))

#Question 26:
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print("Vertical:\n", np.vstack((a,b)))
print("Horizontal:\n", np.hstack((a,b)))

#Question 27:
arr = np.array([[1,2,3],[4,5,6]])
print("Row sum:", arr.sum(axis=1))
print("Col sum:", arr.sum(axis=0))

#Question 28:
arr = np.array([[1, np.nan, 3], [4, 5, np.nan]])
col_mean = np.nanmean(arr, axis=0)
inds = np.where(np.isnan(arr))
arr[inds] = np.take(col_mean, inds[1])
print(arr)

#Question 29:
a = np.array([1,2,3])
b = np.array([4,5,6])
cos_sim = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(cos_sim)

#Question 30:
mat = np.arange(16).reshape(4,4)
rot = np.rot90(mat)
print(rot)

#Question 31:
data = np.array([('Ayan', 20, 85), ('Riya', 21, 90)],
                dtype=[('name','U10'), ('age','i4'), ('marks','f4')])
print(data)
print("Names:", data['name'])
print("Ages:", data['age'])
print("Marks:", data['marks'])
print("Average Marks:", np.mean(data['marks']))

#Question 32:
mat = np.random.rand(3,3)
print(np.linalg.matrix_rank(mat))

#Question 33:
mat = np.random.rand(5,5)
norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
print(norm)

#Question 34:
a = np.array([1,2,3])
b = np.array([1,2,3])
print(np.array_equal(a,b))

#Question 35:
data = np.random.randn(1000)
hist, bins = np.histogram(data, bins=10)
print(hist, bins)

#Question 36:
a = np.array([[1,2,3],[4,5,6]])
b = np.array([10,20,30])
print(a + b)

#Question 37:
arr = np.array([1,2,2,3,3,3,4])
values, counts = np.unique(arr, return_counts=True)
print(values, counts)

#Question 38:
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.corrcoef(a,b))

#Question 39:
arr = np.array([1,2,4,7,11])
grad = np.gradient(arr)
print(grad)

#Question 40:
mat = np.random.rand(3,3)
U, S, V = np.linalg.svd(mat)
print("U:", U, "\nS:", S, "\nV:", V)
