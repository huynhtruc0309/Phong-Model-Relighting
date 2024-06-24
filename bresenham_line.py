def bresenham_line_3d(x0, y0, z0, x1, y1, z1):
    points = []  # List to hold the points on the line

    dx = abs(x1 - x0)  # Absolute difference in x
    dy = abs(y1 - y0)  # Absolute difference in y
    dz = abs(z1 - z0)  # Absolute difference in z

    sx = 1 if x0 < x1 else -1  # Step direction for x
    sy = 1 if y0 < y1 else -1  # Step direction for y
    sz = 1 if z0 < z1 else -1  # Step direction for z

    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        err1 = dy - dx / 2
        err2 = dz - dx / 2

        while x0 != x1:
            points.append((x0, y0, z0))
            if err1 > 0:
                y0 += sy
                err1 -= dx
            if err2 > 0:
                z0 += sz
                err2 -= dx
            err1 += dy
            err2 += dz
            x0 += sx

    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        err1 = dx - dy / 2
        err2 = dz - dy / 2

        while y0 != y1:
            points.append((x0, y0, z0))
            if err1 > 0:
                x0 += sx
                err1 -= dy
            if err2 > 0:
                z0 += sz
                err2 -= dy
            err1 += dx
            err2 += dz
            y0 += sy

    # Driving axis is Z-axis
    else:
        err1 = dx - dz / 2
        err2 = dy - dz / 2

        while z0 != z1:
            points.append((x0, y0, z0))
            if err1 > 0:
                x0 += sx
                err1 -= dz
            if err2 > 0:
                y0 += sy
                err2 -= dz
            err1 += dx
            err2 += dy
            z0 += sz

    points.append((x0, y0, z0))
    return points

# Example usage:
line_points_3d = bresenham_line_3d(93, 163, 244, 8, 163, 96)
print(line_points_3d)
