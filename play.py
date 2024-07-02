def find_z_on_ray(p1, p2, x, y):
    """
    Given two points p1 and p2 defining a ray in 3D, and x, y coordinates,
    find the z coordinate of a point lying on the ray.

    Parameters:
    - p1: Tuple[float, float, float], the first point (x1, y1, z1) on the ray.
    - p2: Tuple[float, float, float], the second point (x2, y2, z2) on the ray.
    - x: float, the x coordinate of the point whose z coordinate is to be found.
    - y: float, the y coordinate of the point whose z coordinate is to be found.

    Returns:
    - float, the z coordinate of the point on the ray.
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # Avoid division by zero if x2 == x1 or y2 == y1
    if x2 != x1:
        t = (x - x1) / (x2 - x1)
    elif y2 != y1:
        t = (y - y1) / (y2 - y1)
    else:
        # The ray is a point or invalid input; cannot determine z uniquely
        raise ValueError("The input points do not define a valid ray.")

    # Calculate the z coordinate using the parameter t
    z = z1 + t * (z2 - z1)

    return z

print(find_z_on_ray((251, 547, 260), (250, 382, 96), 250, 384))