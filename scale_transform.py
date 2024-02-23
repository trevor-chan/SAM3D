import geometry
import numpy as np
import matplotlib.pyplot as plt

def scale_forward (point, shape):
    return point / np.array(shape) * 2 - 1

def scale_backward (point, shape):
    return (point + 1) / 2 * np.array(shape)

def get_intersection_point(point1, point2, idx):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = point2[2] - point1[2]

    if dz == 0:
        return None

    t = (idx - point1[2]) / dz

    if 0 <= t <= 1:
        x_intercept = point1[0] + t * dx
        y_intercept = point1[1] + t * dy
        return (x_intercept, y_intercept, idx)
    else:
        return None

# matrix index to zero centric cube
def index_to_coord (point, transform, shape):
    scaled_point = scale_forward(point, shape)
    transform_point = transform.apply_to_point(scaled_point, inverse=True)
    return transform_point

def coord_to_index (point, transform, shape):
    transform_point = transform.apply_to_point(point)
    scaled_point = scale_backward(transform_point, shape)
    floored_point = np.floor(scaled_point)

    return floored_point

def global_to_local (image, transform):
    return transform.apply_to_array(image)


