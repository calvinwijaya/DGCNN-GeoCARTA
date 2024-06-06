import os
import numpy as np
import laspy
import argparse

output_folder = 'data/sem_seg_data'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process LAS files and save as numpy arrays.")
    parser.add_argument("--las_file", required=False, help="Path to the LAS file.")
    parser.add_argument("--data_folder", required=False, help="Path to the folder containing LAS files.")
    # parser.add_argument("--list_path", required=True, help="Path to the txt file containing a list of LAS files.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # las_file = args.las_file.strip("\"")    
    if args.las_file:
        las_files = [args.las_file]
    elif args.data_folder:
        if os.path.isdir(args.data_folder):
            las_files = [os.path.join(args.data_folder, f) for f in os.listdir(args.data_folder) if f.endswith('.las') or f.endswith('.LAS')]
        else:
            print("Please provide a valid directory path for --data_folder.")
            return
    else:
        print("Please provide either --las_file or --data_folder argument.")
        return

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for idx, las_file in enumerate(las_files):
        out_filename = os.path.join(output_folder, f'Area_{idx + 1}.npy')

        lasfile = laspy.read(las_file)
        x = lasfile.x
        y = lasfile.y
        z = lasfile.z
        r = lasfile.red
        g = lasfile.green
        b = lasfile.blue
        
        # Add Intensity
        I = lasfile.intensity
        
        C = lasfile.classification

        data_label = np.column_stack((x, y, z, r, g, b, I, C))

        xy_min = np.amin(data_label, axis=0)[0:2]
        data_label[:, 0:2] -= xy_min
        data_label[data_label[:, -1] == 6, -1] = 0
        data_label[data_label[:, -1] == 5, -1] = 1

        np.save(out_filename, data_label)

        print(f"Saved {out_filename}")
    
if __name__ == "__main__":
    main()