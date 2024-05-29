import argparse
import numpy as np
import laspy
from osgeo import gdal, gdalconst, osr
import scipy
gdal.UseExceptions()
gdal.DontUseExceptions()

def read_las(input_file):  
    # Read ground point cloud data from the input file
    lasfile = laspy.read(input_file)

    # Extract x, y, z coordinates and classification
    x = lasfile.x
    y = lasfile.y
    z = lasfile.z
    C = lasfile.classification

    # stack data into numpy array
    data = np.column_stack((x, y, z, C))

    return data

def filter_ground(data):
    # Replace column names
    column_names = ['X', 'Y', 'Z', 'C']
    
    # Find the index of the 'Classification' column
    c_column_index = column_names.index('C')
    
    # Filter rows where the 'Classification' column is equal to 1 (Ground)
    data = data[data[:, c_column_index] == 1]

    return data

def create_dem(input_file, output_file, cell_size, type='dtm'):
    # Read point cloud data
    data = read_las(input_file)

    # Filter ground point cloud data
    if type == 'dtm':
        data = filter_ground(data)
    else:
        pass

    # Extract x, y, and z coordinates
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    
    # Define the extent of the DTM
    xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)

    # Create a grid based on the specified cell size
    x_grid = np.arange(xmin, xmax, cell_size)
    y_grid = np.arange(ymin, ymax, cell_size)

    # Create a meshgrid
    X, Y = np.meshgrid(x_grid, y_grid)

    # Binning Minimum Value method
    Z = np.full_like(X, fill_value=np.nan)  # Initialize with NaN
    count = np.zeros_like(X, dtype=int)  # Count of points contributing to each cell
    
    for i in range(len(x)):
        # Find the corresponding grid cell
        col = int((x[i] - xmin) / cell_size)
        row = int((y[i] - ymin) / cell_size)
    
        # Ensure that the point is within the valid range of the grid
        if 0 <= row < Z.shape[0] and 0 <= col < Z.shape[1]:
            # Update the Z value and count for each grid cell
            if np.isnan(Z[row, col]) or z[i] < Z[row, col]:
                Z[row, col] = z[i]
            count[row, col] += 1
    
    # Interpolate NoData values by averaging neighboring cells with multiple iterations
    no_data_mask = np.isnan(Z)
    
    # Number of iterations
    num_iterations = 15
    
    for iteration in range(num_iterations):
        for row in range(Z.shape[0]):
            for col in range(Z.shape[1]):
                if no_data_mask[row, col]:
                    # Define the neighborhood of the current cell, enlarging at each iteration
                    neighborhood = Z[max(0, row - iteration):min(Z.shape[0], row + iteration + 1),
                                      max(0, col - iteration):min(Z.shape[1], col + iteration + 1)]
    
                    # Exclude NaN and NoData values from the neighborhood
                    valid_neighbors = neighborhood[~np.isnan(neighborhood) & ~no_data_mask[max(0, row - iteration):min(Z.shape[0], row + iteration + 1),
                                                                                           max(0, col - iteration):min(Z.shape[1], col + iteration + 1)]]
    
                    # Interpolate the NoData value by averaging valid neighbors
                    if len(valid_neighbors) > 0:
                        Z[row, col] = np.mean(valid_neighbors)
    
    # Optionally, apply minimum value for each grid cell again
    min_elevation = np.nanmin(Z)
    Z[np.isnan(Z)] = min_elevation

    # Apply Gaussian smoothing to the resulting DTM
    sigma = 15.0  # Adjust the standard deviation based on your requirements
    Z = scipy.ndimage.gaussian_filter(Z, sigma=sigma)

    # Set up the GeoTIFF driver
    driver = gdal.GetDriverByName("GTiff")

    # Create the output GeoTIFF file
    dtm_ds = driver.Create(output_file, len(x_grid), len(y_grid), 1, gdalconst.GDT_Float32)

    # Define the spatial reference system
    srs = osr.SpatialReference()
    # srs.ImportFromEPSG(4326)  

    # Set the geotransform (origin and pixel size)
    dtm_ds.SetGeoTransform((xmin, cell_size, 0, ymin, 0, cell_size))

    # Set the spatial reference
    dtm_ds.SetProjection(srs.ExportToWkt())

    # Write the DTM data to the GeoTIFF
    dtm_ds.GetRasterBand(1).WriteArray(Z)

    # Close the GeoTIFF dataset
    dtm_ds = None