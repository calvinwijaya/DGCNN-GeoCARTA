import os
import numpy as np
import laspy
import argparse

output_folder = 'data/sem_seg_data'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process LAS files and save as numpy arrays.")
    parser.add_argument("--las_file", required=False, help="Path to the LAS file.")
    parser.add_argument("--data_folder", required=False, help="Path to the folder containing LAS files.")
    return parser.parse_args()

def main():
    args = parse_arguments()   
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
        C = lasfile.classification

        data_label = np.column_stack((x, y, z, r, g, b, C))

        xy_min = np.amin(data_label, axis=0)[0:2]
        data_label[:, 0:2] -= xy_min
        
        np.save(out_filename, data_label)

        print(f"Saved {out_filename}")
    
if __name__ == "__main__":
    main()