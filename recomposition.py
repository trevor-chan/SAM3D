import numpy as np
import open3d as o3d
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
import matplotlib.pyplot as plt

def create_point_cloud(points, visualize=False, downsample=0, outliers=0, n_neighbors=20, radius=0.02, iterations=1):
    pcd = o3d.geometry.PointCloud()
    points = np.array([p for point in points for p in point])
    pcd.points = o3d.utility.Vector3dVector(points)
    if downsample > 0:
        pcd = pcd.uniform_down_sample(every_k_points=downsample)
    if outliers:
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=n_neighbors, std_ratio=std_ratio)
        for i in range(iterations):
            cl, ind = pcd.remove_radius_outlier(nb_points=n_neighbors, radius=radius)
            pcd = pcd.select_by_index(ind)
    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")
    return pcd
    

def voxel_density_mask(pcd, vox_size = 2/256, resolution=256, dilation=5, erosion=5, fill_holes=True):
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
    dilated = binary_dilation(output, iterations = dilation)
    eroded = binary_erosion(dilated, iterations = erosion)
    mask = binary_fill_holes(eroded)
    return mask

def draw_orthoplanes(image, mask):
    fig, ax = plt.subplots(2,3, figsize=(15,10))
    midpoint = image.shape[0]//2
    ax[0,0].imshow(image[midpoint,:,:], cmap = 'gray')
    ax[0,1].imshow(image[:,midpoint,:], cmap = 'gray')
    ax[0,2].imshow(image[:,:,midpoint], cmap = 'gray')

    ax[1,0].imshow(image[midpoint,:,:], cmap = 'gray')
    ax[1,0].imshow(mask[midpoint,:,:], alpha=0.5, cmap = 'RdBu_r')
    ax[1,1].imshow(image[:,midpoint,:], cmap = 'gray')
    ax[1,1].imshow(mask[:,midpoint,:], alpha=0.5, cmap = 'RdBu_r')
    ax[1,2].imshow(image[:,:,midpoint], cmap = 'gray')
    ax[1,2].imshow(mask[:,:,midpoint], alpha=0.5, cmap = 'RdBu_r')
    plt.show()