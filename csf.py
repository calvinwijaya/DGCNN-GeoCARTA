import argparse
import numpy as np      #pip install numpy
import laspy            #pip install laspy
import CSF              #pip install cloth-simulation-filter
import os

def read_las(input_file):  
    # Read ground point cloud data from the input file
    lasfile = laspy.read(input_file)

    # Extract x, y, z coordinates and r,g, b
    x = lasfile.x
    y = lasfile.y
    z = lasfile.z
    r = lasfile.red
    g = lasfile.green
    b = lasfile.blue

    # stack data into numpy array
    data = np.column_stack((x, y, z, r, g, b))

    return data

def save_las(result, args):
    output_folder = 'output'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    filename = os.path.basename(str(args.input))
    filename = os.path.splitext(filename)[0]
    filename_output = os.path.join(output_folder, filename)

    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)
    las.x = result[:, 0]
    las.y = result[:, 1]
    las.z = result[:, 2]
    las.red = result[:, 3]
    las.green = result[:, 4]
    las.blue = result[:, 5]
    las.classification = result[:, 6]

    las.write(filename_output + ".las")

def csf_filter(args):
    # read las file
    data = read_las(args.input)
    xyz = data[:, :3]

    csf = CSF.CSF()

    # prameter settings
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = args.cr
    # more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/

    csf.setPointCloud(xyz)
    ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
    non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
    csf.do_filtering(ground, non_ground) # do actual filtering.

    # Convert filtering result to array
    ground_arr = np.array(ground)
    non_ground_arr = np.array(non_ground)

    # Create an array for classification with zeros for non-ground points and ones for ground points
    classification_ground = np.ones_like(ground_arr)
    classification_non_ground = np.zeros_like(non_ground_arr)

    # Combine the ground and non-ground classification arrays
    classification = np.concatenate((classification_non_ground, classification_ground), axis=0)

    # Combine the X, Y, Z, R, G, B, and classification arrays for ground and non-ground
    ground_data = np.column_stack((ground_arr, classification_ground))
    non_ground_data = np.column_stack((non_ground_arr, classification_non_ground))

    # Combine the ground and non-ground data arrays
    result = np.concatenate((ground_data, non_ground_data), axis=0)

    # Sort the result array based on the original point IDs
    result = result[result[:, 0].argsort()]

    # Extract the second column (classification) from the result array
    classification_column = result[:, 1]

    # Add the classification column to the original data array
    result = np.column_stack((data, classification_column))

    save_las(result, args)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--input', type=str, default='', required=True, help='specify the point cloud file (LAS)')
    parser.add_argument('--cr', type=float, default=2.0, help='specify Cloth Resolution for CSF')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    csf_filter(args)