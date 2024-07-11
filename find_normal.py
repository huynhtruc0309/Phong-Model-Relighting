import numpy as np

def compute_normal_vector(P1, P2, P3):
    # Convert points to numpy arrays
    P1 = np.array(P1)
    P2 = np.array(P2)
    P3 = np.array(P3)
    
    # Calculate vectors
    v1 = P2 - P1
    v2 = P3 - P1
    
    # Compute the cross product
    N = np.cross(v1, v2)
    
    return N

# Example points
P1 = [224, 65, 41]
P2 = [325, 59, 42]
P3 = [313, 129, 37]

# Calculate the normal vector
normal_vector = compute_normal_vector(P1, P2, P3)
print("Normal Vector:", normal_vector)
