import numpy as np


def point_line_distance(pt, v1, v2):
    """Calculate the distance from a point pt to a line segment v1-v2 """
    line_vec = np.array(v2) - np.array(v1)
    point_vec = np.array(pt) - np.array(v1)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    t = max(0, min(t, 1))
    nearest = np.array(v1) + t * line_vec
    dist = np.linalg.norm(nearest - np.array(pt))
    return dist, nearest


def polygon_circle_distance(polygon, circle_center, radius):
    min_distance = float('inf')
    closest_point = None
    for i in range(len(polygon) - 1):
        p1, p2 = polygon[i], polygon[(i + 1)]
        dist, nearest = point_line_distance(circle_center, p1, p2)
        if dist < min_distance:
            min_distance = dist
            closest_point = nearest
    return min_distance - radius, closest_point
