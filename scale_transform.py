import geometry
import numpy as np
import matplotlib.pyplot as plt

def scale_forward (point, shape):
    return point / np.array(shape) * 2 - 1

def scale_backward (point, shape):
    return (point + 1) / 2 * np.array(shape)

def get_intersection_point(p1, p2, z): #Takes points in global and returns intecept in global if exists, otherwise None

    if (p1[2]-z)*(p2[2]-z) > 0:
        return None
    
    dz = p2[2] - p1[2]
    if dz == 0:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    t = (z - p1[2]) / dz

    x_intercept = p1[0] + t * dx
    y_intercept = p1[1] + t * dy
    return (x_intercept, y_intercept, z)

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


