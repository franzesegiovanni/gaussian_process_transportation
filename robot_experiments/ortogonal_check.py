import numpy as np

def gram_schmidt(matrix):
    Q, _ = np.linalg.qr(matrix)
    return Q

# Example 3x3 matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Orthogonalize the matrix using Gram-Schmidt
orthogonal_matrix = gram_schmidt(A)

print("Original Matrix:")
print(A)
print("\nOrthogonal Matrix:")
print(orthogonal_matrix)
