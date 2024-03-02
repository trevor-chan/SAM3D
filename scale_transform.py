import numpy as np
import cv2
import json

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


def get_prompt_slices(image, dirpath, transform_list):
    slices_list = []
    for a, t in enumerate(transform_list):
        transformed_img = global_to_local(image, t)
        slice_transformed_img = transformed_img[:,:,transformed_img.shape[2]//2]
        cv2.imwrite(f'{dirpath}/slice_{str(a).zfill(2)}.png', slice_transformed_img)

        slice_info = dict()
        slice_info['idx'] = transformed_img.shape[2]//2
        slice_info['transform'] = t
        slice_info['shape'] = transformed_img.shape
        slices_list.append(slice_info)    

    return slices_list

def get_line_segments(slices_list, pos_polylines_slices, neg_polylines_slices):
    pos_seg = []
    neg_seg = []

    for i, s in enumerate(slices_list):
        idx = s['idx']
        shape = s['shape']
        transform_curr = s['transform']

        for line in pos_polylines_slices[i]:
            global_line = []
            for point in line:
                point = point[:2] + [idx]
                print(point)
                transformed_point = index_to_coord(point, transform_curr, shape)
                print(transformed_point)
                global_line.append(transformed_point)
            for j in range(len(global_line) - 1):
                pos_seg.append([global_line[j], global_line[j + 1]])

        for line in neg_polylines_slices[i]:
            global_line = []
            for point in line:
                point = point[:2] + [idx]
                transformed_point = index_to_coord(point, transform_curr, shape)
                global_line.append(transformed_point)
            for j in range(len(global_line) - 1):
                neg_seg.append([global_line[j], global_line[j + 1]])
    return pos_seg, neg_seg

def get_intersections(matrix_shape, pos_seg, neg_seg, t, z):
    pos_intersections = []
    neg_intersections = []

    for p in pos_seg:
        intersection = get_intersection_point(t.apply_to_point(p[0]), t.apply_to_point(p[1]), z)
        if intersection:
            pos_intersections.append(t.apply_to_point(intersection, inverse=True))

    for n in neg_seg:
        intersection = get_intersection_point(t.apply_to_point(n[0]), t.apply_to_point(n[1]), z) # z should be supplied as a global coordinate
        if intersection:
            neg_intersections.append(t.apply_to_point(intersection, inverse=True))

    pos_intersections = [coord_to_index(pt, t, matrix_shape) for pt in pos_intersections]
    neg_intersections = [coord_to_index(pt, t, matrix_shape) for pt in neg_intersections]
    return pos_intersections, neg_intersections

def normalize (image):
    if np.min(image) == np.max(image):
        return np.stack([image, image, image], axis=2)
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    stacked = np.stack([image, image, image], axis=2)
    return stacked

def parse_prompts(folder, slices_list):
    with open(folder + '/prompts.json', 'r') as file:
        prompt_points = json.load(file)
    pos_polylines_slices = []
    neg_polylines_slices = []

    for prompt in prompt_points:
        pos_polylines_slices.append(prompt['pos_polylines'])
        neg_polylines_slices.append(prompt['neg_polylines'])

    # get pos, neg line segments
    pos_seg, neg_seg = get_line_segments(slices_list, pos_polylines_slices, neg_polylines_slices)
    return pos_seg, neg_seg