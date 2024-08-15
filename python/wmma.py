import numpy as np

# Define the 8x8 matrix and the 8x1 vectors
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

# Create the 8x4 matrix B by stacking vectors vertically
# B = np.hstack([v1.reshape(4, 1), v2.reshape(4, 1), v3.reshape(4, 1), v4.reshape(4, 1)])
B = np.zeros((8, 4))
for i in range(0, 8, 4):
    for j in range(0, 4, 1):
        k = 1 + j * (1+i//4)
        B[i:i+4, j] = np.array([k, k, k, k])

# Create the 8x8 matrix C from the 4x4 matrix A
C = np.zeros((8, 8))

# Fill the 8x8 matrix C with blocks of A
C[0:4, 0:4] = A
C[4:8, 4:8] = A

# Perform the matrix-matrix multiplication
result = np.dot(C, B)

# Print the result
print("\nMatrix B:")
print(B)
print("\nMatrix C (block-packed):")
print(C)
print("\nResult of C @ B:")
print(result)
