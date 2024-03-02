import numpy as np
import open3d as o3d
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes




def create_point_cloud(points, visualize=False):
    pcd = o3d.geometry.PointCloud()
    n = np.zeros((len(points), 3))
    for i in range(len(points)):
        n[i] = points[i]
    pcd.points = o3d.utility.Vector3dVector(n)
    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")
    return pcd
    

def voxel_density_mask(pcd, vox_size = 0.01, resolution=64):
    voxgrid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size = vox_size, min_bound = (-1,-1,-1), max_bound = (1,1,1))

    # get binary array
    x = np.linspace(-1, 1, resolution)/3**0.5
    y = np.linspace(-1, 1, resolution)/3**0.5
    z = np.linspace(-1, 1, resolution)/3**0.5
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    queries = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    output = voxgrid.check_if_included(o3d.utility.Vector3dVector(queries))
    output = np.array(output).reshape(resolution,resolution,resolution)
    # mask processing:
    dilated = binary_dilation(output, iterations = 5)
    eroded = binary_erosion(dilated, iterations = 5)
    mask = binary_fill_holes(eroded)
    # mask = output
    return mask