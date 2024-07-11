import numpy as np

# Example points in world coordinates
Q1 = np.array([1, 0, 0])
Q2 = np.array([0, 1, 0])
Q3 = np.array([0, 0, 1])

# Example corresponding points in camera coordinates
P1 = np.array([0.866, -0.5, 0])
P2 = np.array([0.5, 0.866, 0])
P3 = np.array([0, 0, 1])

# Construct matrices Q and P
Q = np.vstack([Q1, Q2, Q3]).T
P = np.vstack([P1, P2, P3]).T

# Calculate the rotation matrix
R = P @ np.linalg.inv(Q)

print("Rotation Matrix R:\n", R)
