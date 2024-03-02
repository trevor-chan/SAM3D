import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance

input_path = ''  # Your input path here
dataname = 'C:\\Users\\aarus\\Downloads\\points_with_noise.txt'  # Your data file name here
max_row = 345376  # Maximum row to read, based on your requirement

# Initialize an empty list to hold the data
data = []

# Open the file and read line by line
with open(input_path + dataname, 'r') as file:
    for i, line in enumerate(file):
        if i == 0:  # Skip the first row if it contains header info
            continue
        if i >= max_row:  # Stop after reading up to the max_row
            break
        # Convert the line to a numpy array and append to the data list
        # Adjust the delimiter as necessary, here assuming whitespace
        data.append(np.fromstring(line, sep=' '))

# Convert the list of arrays into a single 2D numpy array
sphere = []
size = 100
sphereinit = np.zeros((size,3))
for i in range(0,size):
    for j in range(0,size):
        for k in range(0,size):
            if (i-size/2)**2 + (j-size/2)**2 + (k-size/2)**2 <= (20+5*np.random.uniform(-1, 1))**2:
                sphere.append((i,j,k))

sphere = np.array(sphere)




pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
pcd.points = o3d.utility.Vector3dVector(sphere)
o3d.visualization.draw_geometries([pcd], window_name="Triangle Mesh")

# pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255)
# pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,6:9])
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Optionally, you might want to orient the normals in a consistent manner, especially for visualization
pcd.orient_normals_consistent_tangent_plane(k=10)

from scipy.stats import gaussian_kde

def remove_low_density_points(points):
    """
    Remove points with bottom 10% density using Gaussian KDE.
    
    Parameters:
    - points: A numpy array of shape (n, 3), representing 3D points.
    
    Returns:
    - filtered_points: A numpy array of the points with top 90% density.
    """
    # Calculate the Gaussian KDE for each point

    kde = gaussian_kde(points.T)  # Transpose points to shape (3, n) for kde


    # Evaluate the densities at the locations of the points
    # densities = kde(points.T)  # Use transposed points as input
    X,Y,Z = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
    coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    # print("done3")
    print("finding densities")
    densities = kde(coords)
    print("densities found")
    # densities = np.reshape(densities, X.shape)
    density_threshold = np.percentile(densities, 10)
    print(densities.shape)
    print("done2")
    print(density_threshold)
    # densities = np.where(densities > density_threshold, 1, 0)
    density_mask = densities > density_threshold
    coords = coords.T

    high_density_grid_points = coords[density_mask]

    # return densitycoords
    return high_density_grid_points


# sigma = 0.1
# threshold = 0.9
# grid_size = 50


filtered_grid_points = remove_low_density_points(sphere)
print("shape = ", filtered_grid_points.shape)
print(sphere.shape)
# print(filtered_grid.shape)
# print(filtered_grid)
filtered_grid = o3d.geometry.PointCloud()
filtered_grid.points = o3d.utility.Vector3dVector(filtered_grid_points)
o3d.visualization.draw_geometries([filtered_grid], window_name="Point Cloud1")
print("done1")

import numpy as np
from scipy.stats import gaussian_kde

def create_density_mask(points, resolution=1000, threshold_percentile=80):
    """
    Create a 3D density mask from points using Gaussian KDE.
    
    Parameters:
    - points: A numpy array of shape (n, 3), representing 3D points.
    - resolution: The desired resolution for each dimension of the output grid.
    - threshold_percentile: The percentile for density thresholding to create the mask.
    
    Returns:
    - mask: A numpy array of shape (resolution, resolution, resolution), representing the 3D binary mask.
    """
    # Calculate the Gaussian KDE for the input points
    kde = gaussian_kde(points.T)  # Transpose points to shape (3, n) for KDE
    
    # Generate a grid of new points within the range [-1, +1] for each dimension
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Evaluate the KDE on the grid points to calculate densities
    densities = kde(grid_points).reshape(resolution, resolution, resolution)
    
    # Threshold the density array to create a 3D binary mask
    density_threshold = np.percentile(densities, threshold_percentile)
    mask = np.where(densities > density_threshold, 1, 0)
    
    return mask

mask = create_density_mask(sphere, resolution=25, threshold_percentile=10)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt

def visualize_3d_mask(mask):
    fig = plt.figure()
    ax = fig.add_subplot(1111, projection='3d')
    
    # Generate the voxel coordinates
    x, y, z = np.indices(np.array(mask.shape) + 1)
    ax.voxels(x, y, z, mask, facecolors='blue', edgecolor='k')
    
    plt.show()


# Assuming 'mask' is your 3D binary mask array from the previous function
# visualize_3d_mask(mask)

visualize_3d_mask(mask)

# o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

# Radius determination
# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# radius = 10* avg_dist

# bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

# dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

# dec_mesh.remove_degenerate_triangles()
# dec_mesh.remove_duplicated_triangles()
# dec_mesh.remove_duplicated_vertices()
# dec_mesh.remove_non_manifold_edges()

# o3d.visualization.draw_geometries([dec_mesh], window_name="Triangle Mesh")

filtered_grid.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Optionally, you might want to orient the normals in a consistent manner, especially for visualization
filtered_grid.orient_normals_consistent_tangent_plane(k=10)

poisson_mesh =o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(filtered_grid,depth=3, width=0, scale=1.1, linear_fit=False)[0]


print("done2")


bbox = filtered_grid.get_axis_aligned_bounding_box()
print("done3")
p_mesh_crop = poisson_mesh.crop(bbox)
print("done4")
o3d.visualization.draw_geometries([p_mesh_crop], window_name="Triangle Mesh")
print("done5")
