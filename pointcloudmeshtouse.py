import numpy as np
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


def create_density_mask(points, resolution=10, threshold_percentile=80):
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
    print("finding densities")

    densities = kde(grid_points).reshape(resolution, resolution, resolution)
    print("densities found")

    
    # Threshold the density array to create a 3D binary mask
    density_threshold = np.percentile(densities, threshold_percentile)
    mask = np.where(densities > density_threshold, 1, 0)

    print("mask found")
    
    return mask

# mask = create_density_mask(sphere, resolution=25, threshold_percentile=10)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt

def visualize_3d_mask(mask):
    print("visualizing")
    fig = plt.figure()
    ax = fig.add_subplot(1111, projection='3d')
    
    # Generate the voxel coordinates
    x, y, z = np.indices(np.array(mask.shape) + 1)
    ax.voxels(x, y, z, mask, facecolors='blue', edgecolor='k')
    
    plt.show()

import numpy as np
import fastkde

def kde(points, resolution=10, threshold_percentile=80):
    """
    Create a 3D density mask from points using fastKDE.
    
    Parameters:
    - points: A numpy array of shape (n, 3), representing 3D points.
    - resolution: The desired resolution for each dimension of the output grid.
    - threshold_percentile: The percentile for density thresholding to create the mask.
    
    Returns:
    - mask: A numpy array of shape (resolution, resolution, resolution), representing the 3D binary mask.
    """
    # Calculate the KDE for the input points using fastKDE
    # myPDF = fastKDE.pdf(points[:,0], points[:,1], points[:,2])

    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    print(grid_points.shape)
    print(points.shape)

    test_point_pdf_values = fastkde.pdf_at_points(points[:,0], points[:,1], points[:,2], list_of_points = grid_points)

    # densities = myPDF(grid_points).reshape(resolution, resolution, resolution)
    print("densities found")

    
    # Threshold the density array to create a 3D binary mask
    density_threshold = np.percentile(test_point_pdf_values, threshold_percentile)
    mask = np.where(test_point_pdf_values > density_threshold, 1, 0)
    
    return mask
