import numpy as np
import scipy
import geometry

def rot_from_vecs(vec1, vec2=[1,0,0]):
    matrix = rotation_matrix_from_vectors(vec1, vec2)
    transform = geometry.Transform(translation=[0,0,0], rotation=scipy.spatial.transform.Rotation.from_matrix(matrix).as_euler('zyx', degrees=False))
    return transform
    
    
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def get_dodecahedron_transforms():
    gr = (1 + np.sqrt(5))/2
    dodecahedron_vertices = np.array([
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],[-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    [0, gr, 1 / gr], [0, -gr, 1 / gr], [0, gr, -1 / gr], [0, -gr, -1 / gr],
    [gr, 1 / gr, 0], [-gr, 1 / gr, 0], [gr, -1 / gr, 0], [-gr, -1 / gr, 0],
    [1 / gr, 0, gr], [1 / gr, 0, -gr], [-1 / gr, 0, gr], [-1 / gr, 0, -gr]])
    transforms = []
    for vec in dodecahedron_vertices:
        transforms.append(rot_from_vecs(vec))
    return transforms

def get_icosahedron_transforms():
    gr = (1 + np.sqrt(5))/2
    icosahedron_vertices = np.array([[0,1,gr],[0,1,-gr],[0,-1,gr],[0,-1,-gr],
                                     [1,gr,0],[1,-gr,0],[-1,gr,0],[-1,-gr,0],
                                     [gr,0,1],[-gr,0,1],[gr,0,-1],[-gr,0,-1]])
    transforms = []
    for vec in icosahedron_vertices:
        transforms.append(rot_from_vecs(vec))
    return transforms

def get_cube_transforms():
    cube_vertices = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1],
                          [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    transforms = []
    for vec in cube_vertices:
        transforms.append(rot_from_vecs(vec))
    return transforms